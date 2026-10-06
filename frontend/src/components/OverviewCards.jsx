export default function OverviewCards({ overview }) {
  if (!overview) return null;

  const up = overview.pct_change_vs_prior_30d >= 0;

  return (
    <div className="kpi-grid">
      <div className="card">
        <div className="kpi-label">Stores Tracked</div>
        <div className="kpi-value">{overview.num_stores}</div>
      </div>
      <div className="card">
        <div className="kpi-label">Sales — Last 30 Days</div>
        <div className="kpi-value">${Number(overview.total_sales_last_30d).toLocaleString()}</div>
        <div className={`kpi-delta ${up ? "up" : "down"}`}>
          {up ? "▲" : "▼"} {Math.abs(overview.pct_change_vs_prior_30d)}% vs prior 30d
        </div>
      </div>
      <div className="card">
        <div className="kpi-label">Avg Daily Sales / Store</div>
        <div className="kpi-value">${Number(overview.avg_daily_sales_per_store).toLocaleString()}</div>
      </div>
      <div className="card">
        <div className="kpi-label">Latest Data Point</div>
        <div className="kpi-value" style={{ fontSize: 20 }}>{overview.last_date}</div>
      </div>
    </div>
  );
}
