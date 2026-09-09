import base64
import logging
import os

from opentelemetry import metrics, trace

_logger = logging.getLogger(__name__)

_tracer_provider = None
_meter_provider = None


def _exporter_target():
    """Where to send telemetry, and what auth it needs.

    Two supported shapes, checked in order:

      * OTEL_EXPORTER_OTLP_ENDPOINT -- an OTLP collector we already trust.
        No auth: in-cluster the hop never leaves the node, and the collector
        is the only thing holding the Grafana credentials.
      * GRAFANA_OTLP_ENDPOINT + GRAFANA_INSTANCE_ID + GRAFANA_OTLP_TOKEN --
        straight to Grafana Cloud over HTTP Basic. This is the Lambda path.

    The Basic header is built here rather than via OTEL_EXPORTER_OTLP_HEADERS
    so we avoid that variable's space-encoding trap.

    Returns (base_url, headers) with no trailing slash on base_url, or None
    if telemetry isn't configured at all.
    """
    collector = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if collector:
        return collector.rstrip("/"), {}

    endpoint = os.environ.get("GRAFANA_OTLP_ENDPOINT")
    instance = os.environ.get("GRAFANA_INSTANCE_ID")
    token = os.environ.get("GRAFANA_OTLP_TOKEN")
    # All three, not two: endpoint+token with no instance id used to pass the
    # check and then raise KeyError at import time, killing the container.
    if endpoint and instance and token:
        creds = base64.b64encode(f"{instance}:{token}".encode()).decode()
        return endpoint.rstrip("/"), {"Authorization": f"Basic {creds}"}

    return None


def _resource_attributes():
    """Resource attributes for this process.

    The cloud/faas keys are Lambda-specific and are only added when we're
    actually on Lambda -- AWS_LAMBDA_FUNCTION_NAME is always set there. In
    Kubernetes they're absent, and the collector's k8sattributes processor
    supplies k8s.pod.name / k8s.node.name / k8s.deployment.name instead.

    Previously cloud.provider was hardcoded to "aws" and cloud.region and
    faas.name defaulted to "", which exported empty-string attributes on a
    workload that isn't a function.
    """
    attrs = {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "embedvisual"),
        "deployment.environment": os.environ.get("ENVIRONMENT", "production"),
    }

    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if function_name:
        attrs["cloud.provider"] = "aws"
        attrs["cloud.region"] = os.environ.get("AWS_REGION", "")
        attrs["faas.name"] = function_name

    return attrs


def _configure_exporters(base, headers):
    """Wire up real OTLP exporters for traces and metrics.

    `base` is an OTLP base URL without a signal path; `headers` may be empty.
    """
    global _tracer_provider, _meter_provider

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import (
        ExplicitBucketHistogramAggregation,
        View,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(_resource_attributes())

    # --- Traces ---
    # The proto-http exporter uses the endpoint you pass verbatim, so we must
    # append the signal-specific path ourselves (/v1/traces, /v1/metrics).
    # (The collector's own otlphttp exporter appends them for you -- opposite
    # convention, same base URL works for both.)
    # timeout is capped so a network hiccup can't hang the request forever.
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{base}/v1/traces", headers=headers, timeout=5)
        )
    )
    trace.set_tracer_provider(_tracer_provider)

    # --- Metrics ---
    # Durations are recorded in SECONDS, but OTel's default histogram buckets
    # (0, 5, 10, 25, ... 10000) are meant for milliseconds. Left as-is, every
    # sub-5s request lands in the first [0, 5] bucket and histogram_quantile
    # just returns 5 * quantile (p50->2.5, p95->4.75). These views override the
    # buckets with second-scale boundaries: dense from 5ms to 1s where warm
    # invocations live, with headroom to 10s for cold starts.
    latency_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    duration_views = [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(
                boundaries=latency_buckets
            ),
        )
        for name in ("embed.duration", "embed.inference.duration")
    ]

    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=f"{base}/v1/metrics", headers=headers, timeout=5
                )
            )
        ],
        views=duration_views,
    )
    metrics.set_meter_provider(_meter_provider)


# Only configure real exporters if we actually have somewhere to send data.
# Otherwise everything below stays a harmless no-op -- but say so out loud,
# because silently no-op telemetry is the hardest version of this to diagnose.
_target = _exporter_target()
if _target is not None:
    _configure_exporters(*_target)
else:
    _logger.warning(
        "otel: no OTLP endpoint configured (set OTEL_EXPORTER_OTLP_ENDPOINT, "
        "or GRAFANA_OTLP_ENDPOINT + GRAFANA_INSTANCE_ID + GRAFANA_OTLP_TOKEN); "
        "telemetry is disabled"
    )

# tracer / instruments work whether or not a provider was set: with no provider
# configured, the OTel API hands back no-op objects that safely do nothing.
tracer = trace.get_tracer("embedvisual")
_meter = metrics.get_meter("embedvisual")

request_counter = _meter.create_counter(
    "embed.requests", unit="1", description="Number of embedding requests"
)
duration_hist = _meter.create_histogram(
    "embed.duration", unit="s", description="End-to-end handler duration"
)
inference_hist = _meter.create_histogram(
    "embed.inference.duration", unit="s", description="ONNX inference duration"
)
batch_hist = _meter.create_histogram(
    "embed.batch.size", unit="1", description="Number of texts per request"
)


def flush():
    """Force-export any buffered telemetry BEFORE the handler returns.

    Lambda freezes the execution environment as soon as the handler finishes,
    so the SDK's normal background export would never run. Call this at the
    end of every invocation (e.g. in a finally block)."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush(5000)  # milliseconds
    if _meter_provider is not None:
        _meter_provider.force_flush(5000)