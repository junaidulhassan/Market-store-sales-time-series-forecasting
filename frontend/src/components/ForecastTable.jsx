export default function ForecastTable({ forecast }) {
  if (!forecast?.length) return null;
  return (
    <div className="table-wrap">
      <table className="log-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>p10</th>
            <th>Median (p50)</th>
            <th>p90</th>
          </tr>
        </thead>
        <tbody>
          {forecast.map((f) => (
            <tr key={f.date}>
              <td>{f.date}</td>
              <td>${f.p10.toLocaleString()}</td>
              <td style={{ color: "#ffb84d", fontWeight: 700 }}>${f.p50.toLocaleString()}</td>
              <td>${f.p90.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
