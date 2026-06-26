"""
UDP beacon for zero-config peer discovery.

Each node broadcasts its presence (node_id, ip, grpc_port) every BEACON_INTERVAL
seconds to the subnet broadcast address. Any node hearing a new beacon calls the
on_peer_discovered callback so the transport can add the peer immediately.

The broadcast address is computed dynamically from the actual network interface,
so this works regardless of whether the subnet is /18, /24, or anything else.
"""

import fcntl
import json
import logging
import socket
import struct
import threading

logger = logging.getLogger(__name__)

BEACON_PORT = 50052
BEACON_INTERVAL = 10  # seconds between broadcasts

# Linux ioctl codes
_SIOCGIFADDR    = 0x8915
_SIOCGIFBRDADDR = 0x8919


def _get_broadcast_address(my_ip: str) -> str:
    """
    Compute the directed broadcast for the interface that owns my_ip.
    Uses only stdlib fcntl — no external commands needed inside Docker.
    """
    try:
        # Read all interface names from /proc/net/dev (always available on Linux)
        ifaces = []
        with open("/proc/net/dev") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    ifaces.append(line.split(":")[0].strip())

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for iface in ifaces:
                try:
                    ifreq = struct.pack("16sH14s", iface.encode()[:15], socket.AF_INET, b"\x00" * 14)
                    res = fcntl.ioctl(s.fileno(), _SIOCGIFADDR, ifreq)
                    if socket.inet_ntoa(res[20:24]) == my_ip:
                        res = fcntl.ioctl(s.fileno(), _SIOCGIFBRDADDR, ifreq)
                        return socket.inet_ntoa(res[20:24])
                except OSError:
                    continue
    except Exception as e:
        logger.warning(f"Could not compute broadcast address for {my_ip}: {e}")
    return "255.255.255.255"


class BeaconService:
    """
    Runs two daemon threads: one that broadcasts this node's presence, one that
    listens for peers and fires on_peer_discovered(peer_id, "ip:port") for each
    newly seen node.
    """

    def __init__(
        self,
        node_id: str,
        my_ip: str,
        grpc_port: int,
        on_peer_discovered,
    ):
        self.node_id = node_id
        self.my_ip = my_ip
        self.grpc_port = grpc_port
        self.on_peer_discovered = on_peer_discovered
        self.broadcast_addr = _get_broadcast_address(my_ip)
        self._stop = threading.Event()
        self._known_peers: set = set()

        logger.info(
            f"[{node_id}] BeaconService: broadcast target {self.broadcast_addr}:{BEACON_PORT}"
        )

    def start(self):
        threading.Thread(target=self._broadcaster, daemon=True, name="beacon-tx").start()
        threading.Thread(target=self._listener, daemon=True, name="beacon-rx").start()

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _broadcaster(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        payload = json.dumps({
            "node_id": self.node_id,
            "ip": self.my_ip,
            "port": self.grpc_port,
        }).encode()
        try:
            while not self._stop.is_set():
                try:
                    sock.sendto(payload, (self.broadcast_addr, BEACON_PORT))
                except Exception as e:
                    logger.warning(f"[{self.node_id}] Beacon broadcast error: {e}")
                self._stop.wait(BEACON_INTERVAL)
        finally:
            sock.close()

    def _listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)
        sock.bind(("", BEACON_PORT))
        try:
            while not self._stop.is_set():
                try:
                    data, _ = sock.recvfrom(1024)
                    self._handle(data)
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.warning(f"[{self.node_id}] Beacon receive error: {e}")
        finally:
            sock.close()

    def _handle(self, data: bytes):
        try:
            info = json.loads(data.decode())
            peer_id = info["node_id"]
            peer_ip = info["ip"]
            peer_port = int(info["port"])
        except Exception:
            return

        if peer_ip == self.my_ip:
            return  # ignore own beacon

        if peer_id not in self._known_peers:
            self._known_peers.add(peer_id)
            logger.info(
                f"[{self.node_id}] Beacon: discovered new peer {peer_id} @ {peer_ip}:{peer_port}"
            )
            self.on_peer_discovered(peer_id, f"{peer_ip}:{peer_port}")
