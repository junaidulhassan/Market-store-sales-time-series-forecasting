import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const GREEN = "#00e676";
const TEAL = "#00d1a0";
const AMBER = "#ffb84d";
const GRID = "#1d2b25";

function buildChartData(history, forecast) {
  const historyPoints = history.map((h) => ({
    date: h.date,
    actual: h.sales,
  }));

  const forecastPoints = forecast.map((f) => ({
    date: f.date,
    forecast: f.p50,
    band: [f.p10, f.p90],
  }));

  return [...historyPoints, ...forecastPoints];
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const actual = payload.find((p) => p.dataKey === "actual");
  const forecast = payload.find((p) => p.dataKey === "forecast");
  return (
    <div
      style={{
        background: "#0e1815",
        border: "1px solid rgba(0,230,118,0.3)",
        borderRadius: 10,
        padding: "10px 14px",
        fontSize: 12.5,
        color: "#e9f5ef",
      }}
    >
      <div style={{ color: "#8fa99e", marginBottom: 6 }}>{label}</div>
      {actual && <div>Actual: <b style={{ color: TEAL }}>${actual.value?.toFixed(0)}</b></div>}
      {forecast && <div>Forecast (median): <b style={{ color: AMBER }}>${forecast.value?.toFixed(0)}</b></div>}
    </div>
  );
}

export default function SalesChart({ history, forecast }) {
  const data = buildChartData(history, forecast);
  const splitDate = forecast?.[0]?.date;

  return (
    <div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={TEAL} stopOpacity={0.35} />
              <stop offset="100%" stopColor={TEAL} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: "#8fa99e", fontSize: 11 }}
            minTickGap={25}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#8fa99e", fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            width={60}
          />
          <Tooltip content={<CustomTooltip />} />
          {splitDate && (
            <ReferenceLine x={splitDate} stroke={AMBER} strokeDasharray="4 4" strokeOpacity={0.6} />
          )}
          <Area
            dataKey="band"
            stroke="none"
            fill={AMBER}
            fillOpacity={0.12}
            isAnimationActive={false}
            connectNulls
          />
          <Area
            type="monotone"
            dataKey="actual"
            stroke={TEAL}
            strokeWidth={2}
            fill="url(#actualFill)"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke={AMBER}
            strokeWidth={2.5}
            strokeDasharray="6 3"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="legend-row">
        <div className="legend-item">
          <span className="legend-swatch" style={{ background: TEAL }} />
          Historical sales
        </div>
        <div className="legend-item">
          <span className="legend-swatch" style={{ background: AMBER }} />
          Forecast (median)
        </div>
        <div className="legend-item">
          <span className="legend-swatch" style={{ background: "rgba(255,184,77,0.3)" }} />
          p10–p90 forecast range
        </div>
      </div>
    </div>
  );
}
