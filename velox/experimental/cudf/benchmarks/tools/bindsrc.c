// Binds every outbound TCP connection to a fixed local source address, so that
// one process's S3 traffic leaves through one NIC.
//
// KvikIO never exposes libcurl's CURLOPT_INTERFACE, so there is no in-process
// way to choose an egress NIC. This host already carries `ip rule` entries that
// send each secondary NIC's source address out its own route table, so fixing
// the source address is enough to pick the NIC, and it needs no capability
// beyond what an unprivileged container already has.
//
//   gcc -shared -fPIC -O2 -o libbindsrc.so bindsrc.c -ldl
//   BIND_SRC_IP=172.31.241.193 LD_PRELOAD=./libbindsrc.so <cmd>
//
// Leaving BIND_SRC_IP unset makes the shim a no-op, so the same wrapper can run
// an unpinned control.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef IP_BIND_ADDRESS_NO_PORT
#define IP_BIND_ADDRESS_NO_PORT 24
#endif

typedef int (*connect_fn)(int, const struct sockaddr *, socklen_t);

#define MAX_DST 64

static connect_fn real_connect;
static struct in_addr bind_addr;
static int bind_enabled;

// Destination fan-out. glibc hands libcurl every /etc/hosts entry for the S3
// name, but libcurl connects to the first that answers and KvikIO never sets
// CURLOPT_DNS_SHUFFLE_ADDRESSES, so every connection lands on one front end.
// Spreading them is what turns a pool of many transfers into many independent
// S3 endpoints. SNI and the Host header still carry the hostname, so
// certificate verification is unaffected.
static struct in_addr dst_addrs[MAX_DST];
static int dst_count;
static unsigned long dst_next;
static unsigned short dst_port;
static int debug;

__attribute__((constructor)) static void bindsrc_init(void) {
  real_connect = (connect_fn)dlsym(RTLD_NEXT, "connect");

  const char *ip = getenv("BIND_SRC_IP");
  if (ip != NULL && ip[0] != '\0' && inet_pton(AF_INET, ip, &bind_addr) == 1) {
    bind_enabled = 1;
  }

  // Only rewrite connections to this port, so nothing else the process does
  // (IMDS, DNS) is touched.
  const char *port = getenv("FANOUT_DST_PORT");
  dst_port = (unsigned short)((port != NULL && port[0] != '\0') ? atoi(port) : 443);

  const char *list = getenv("FANOUT_DST_IPS");
  if (list != NULL && list[0] != '\0') {
    char buf[MAX_DST * 16];
    strncpy(buf, list, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (char *tok = strtok(buf, ","); tok != NULL && dst_count < MAX_DST;
         tok = strtok(NULL, ",")) {
      if (inet_pton(AF_INET, tok, &dst_addrs[dst_count]) == 1) { dst_count++; }
    }
  }

  debug = getenv("BINDSRC_DEBUG") != NULL;
  if (debug) {
    fprintf(stderr, "[bindsrc] pid=%d src=%s dst_count=%d port=%u\n", getpid(),
            bind_enabled ? getenv("BIND_SRC_IP") : "(none)", dst_count,
            (unsigned)dst_port);
  }
}

int connect(int fd, const struct sockaddr *addr, socklen_t addrlen) {
  struct sockaddr_in rewritten;

  if (real_connect == NULL) {
    real_connect = (connect_fn)dlsym(RTLD_NEXT, "connect");
  }

  if (bind_enabled && addr != NULL && addr->sa_family == AF_INET) {
    int type = 0;
    socklen_t type_len = sizeof(type);

    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_len) == 0 &&
        type == SOCK_STREAM) {
      struct sockaddr_in local;
      int on = 1;

      memset(&local, 0, sizeof(local));
      local.sin_family = AF_INET;
      local.sin_addr = bind_addr;
      local.sin_port = 0;

      // Defer port selection to connect(), which lets thousands of concurrent
      // connections share the ephemeral range keyed on the full 4-tuple. A
      // plain bind() to port 0 picks the port up front and runs out far sooner.
      setsockopt(fd, IPPROTO_IP, IP_BIND_ADDRESS_NO_PORT, &on, sizeof(on));

      // Best effort: a socket the caller already bound fails here, and that
      // caller's choice should win rather than break the connection.
      bind(fd, (const struct sockaddr *)&local, sizeof(local));
    }
  }

  if (dst_count > 0 && addr != NULL && addr->sa_family == AF_INET &&
      addrlen >= (socklen_t)sizeof(struct sockaddr_in)) {
    const struct sockaddr_in *want = (const struct sockaddr_in *)addr;

    if (ntohs(want->sin_port) == dst_port) {
      unsigned long slot = __atomic_fetch_add(&dst_next, 1, __ATOMIC_RELAXED);

      memcpy(&rewritten, want, sizeof(rewritten));
      rewritten.sin_addr = dst_addrs[slot % (unsigned long)dst_count];
      if (debug) {
        char a[INET_ADDRSTRLEN], b[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &want->sin_addr, a, sizeof(a));
        inet_ntop(AF_INET, &rewritten.sin_addr, b, sizeof(b));
        fprintf(stderr, "[bindsrc] rewrite %s -> %s\n", a, b);
      }
      return real_connect(fd, (const struct sockaddr *)&rewritten,
                          sizeof(rewritten));
    }
  }

  return real_connect(fd, addr, addrlen);
}
