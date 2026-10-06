import { useMemo, useState } from "react";

export default function Sidebar({ stores, selectedStore, onSelectStore }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!query.trim()) return stores;
    const q = query.toLowerCase();
    return stores.filter(
      (s) =>
        String(s.store_nbr).includes(q) ||
        s.city.toLowerCase().includes(q) ||
        s.state.toLowerCase().includes(q)
    );
  }, [stores, query]);

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">SF</div>
        <div>
          <div className="brand-title">Sales Forecaster</div>
          <div className="brand-subtitle">Transformer-powered demand AI</div>
        </div>
      </div>

      <input
        className="store-search"
        placeholder="Search store, city or state..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className="store-list">
        {filtered.map((s) => (
          <div
            key={s.store_nbr}
            className={`store-item ${selectedStore === s.store_nbr ? "active" : ""}`}
            onClick={() => onSelectStore(s.store_nbr)}
          >
            <div>
              <div className="store-item-name">Store #{s.store_nbr}</div>
              <div className="store-item-meta">{s.city}, {s.state}</div>
            </div>
            <span className="store-badge">{s.type}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}
