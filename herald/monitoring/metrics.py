import time
from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self):
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1, **labels: str) -> None:
        with self._lock:
            self._counters[(name, self._labels(labels))] += value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[(name, self._labels(labels))] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._histograms[(name, self._labels(labels))].append(value)

    def render_prometheus(self) -> str:
        lines = [
            "# HELP herald_build_info Static HERALD service metadata.",
            "# TYPE herald_build_info gauge",
            'herald_build_info{service="api"} 1',
        ]

        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{self._format_labels(labels)} {value}")

            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{self._format_labels(labels)} {value}")

            for (name, labels), values in sorted(self._histograms.items()):
                count = len(values)
                total = sum(values)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{self._format_labels(labels)} {count}")
                lines.append(f"{name}_sum{self._format_labels(labels)} {total}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _labels(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key, str(value)) for key, value in labels.items()))

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        formatted = ",".join(f'{key}="{value}"' for key, value in labels)
        return f"{{{formatted}}}"


metrics = MetricsRegistry()


class Timer:
    def __init__(self, metric_name: str, **labels: str):
        self.metric_name = metric_name
        self.labels = labels
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, traceback):
        metrics.observe(self.metric_name, time.monotonic() - self.started_at, **self.labels)
