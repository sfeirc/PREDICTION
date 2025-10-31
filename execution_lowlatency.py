"""
Low-latency execution adapter (microsecond-level, co-location ready).

This module defines an interface that can be wired to a native gateway (C/C++/FPGA)
via ctypes/cffi/pybind11. The default implementation is a no-op simulator
with microsecond timestamps, so the rest of the bot can exercise the path.
"""

import time
import ctypes
from typing import Dict, Optional

try:
    perf_counter_ns = time.perf_counter_ns
except AttributeError:
    def perf_counter_ns():
        return int(time.perf_counter() * 1e9)


class LowLatencyExecutor:
    def __init__(self, venue: str = "binance", microseconds_target: int = 1000, colocated: bool = False):
        self.venue = venue
        self.us_target = int(microseconds_target)
        self.colocated = bool(colocated)
        self._gateway = None  # Placeholder for native handle

    # --- Lifecycle ---
    def warm_up(self) -> None:
        # Pre-allocate objects, pin threads/affinity if needed, load native lib
        # Example: self._gateway = ctypes.CDLL("gateway.dll")
        pass

    def time_sync(self, method: str = "ptp") -> None:
        # Hook to ensure synchronized clock (PTP/NTP) with hardware timestamping
        pass

    # --- Order API (microsecond stamps) ---
    def _stamp_us(self) -> int:
        return perf_counter_ns() // 1000

    def send_order(self, symbol: str, side: str, qty: float, price: Optional[float] = None, tif: str = "IOC") -> Dict:
        ts_send_us = self._stamp_us()
        # In production, call native gateway here; this is a fast-path mock
        ts_ack_us = self._stamp_us()
        return {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "tif": tif,
            "ts_send_us": ts_send_us,
            "ts_ack_us": ts_ack_us,
            "latency_us": max(0, ts_ack_us - ts_send_us),
            "status": "ACK"
        }

    def cancel_order(self, order_id: str) -> Dict:
        ts_send_us = self._stamp_us()
        ts_ack_us = self._stamp_us()
        return {"order_id": order_id, "ts_send_us": ts_send_us, "ts_ack_us": ts_ack_us, "status": "CANCELLED"}

    # --- Market data fast-path (optional wiring) ---
    def on_md_tick(self, tick: Dict) -> None:
        # Hook for fast L2/L3 events if using native feed handlers
        pass

    # --- Risk checks (pre-trade) ---
    def pre_trade_checks(self, notional: float, max_heat: float = 0.06) -> bool:
        # Placeholders for portfolio heat / limits
        return True
