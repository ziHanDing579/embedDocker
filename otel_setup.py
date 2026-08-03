import base64
import os

from opentelemetry import metrics, trace

_tracer_provider = None
_meter_provider = None


def _basic_auth_header():
    """Build the HTTP Basic auth header Grafana Cloud expects:
    Authorization: Basic base64(instance_id:token)
    Built in code so we avoid the OTEL_EXPORTER_OTLP_HEADERS space-encoding trap."""
    instance = os.environ["GRAFANA_INSTANCE_ID"]
    token = os.environ["GRAFANA_OTLP_TOKEN"]
    creds = base64.b64encode(f"{instance}:{token}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _configure_exporters():
    """Wire up real OTLP exporters for traces and metrics -> Grafana Cloud."""
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

    base = os.environ["GRAFANA_OTLP_ENDPOINT"].rstrip("/")
    headers = _basic_auth_header()

    resource = Resource.create(
        {
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "embedvisual"),
            "deployment.environment": os.environ.get("ENVIRONMENT", "production"),
            "cloud.provider": "aws",
            "cloud.region": os.environ.get("AWS_REGION", ""),
            "faas.name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", ""),
        }
    )

    # --- Traces ---
    # The proto-http exporter uses the endpoint you pass verbatim, so we must
    # append the signal-specific path ourselves (/v1/traces, /v1/metrics).
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
# Otherwise everything below stays a harmless no-op.
if os.environ.get("GRAFANA_OTLP_ENDPOINT") and os.environ.get("GRAFANA_OTLP_TOKEN"):
    _configure_exporters()

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