import { useEffect, useState, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import OverviewCards from "./components/OverviewCards";
import SalesChart from "./components/SalesChart";
import AddDataForm from "./components/AddDataForm";
import ForecastTable from "./components/ForecastTable";
import { getStores, getOverview, getHistory, getForecast } from "./api";

export default function App() {
  const [stores, setStores] = useState([]);
  const [overview, setOverview] = useState(null);
  const [selectedStore, setSelectedStore] = useState(null);
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getStores().then((data) => {
      setStores(data);
      if (data.length) setSelectedStore(data[0].store_nbr);
    }).catch(() => setError("Could not reach the API. Is the backend running on port 8000?"));
    getOverview().then(setOverview).catch(() => {});
  }, []);

  const loadStoreData = useCallback((storeNbr) => {
    if (!storeNbr) return;
    setLoadingForecast(true);
    setError(null);
    Promise.all([getHistory(storeNbr, 120), getForecast(storeNbr)])
      .then(([h, f]) => {
        setHistory(h);
        setForecast(f);
      })
      .catch(() => setError("Failed to load store data from the API."))
      .finally(() => setLoadingForecast(false));
  }, []);

  useEffect(() => {
    if (selectedStore) loadStoreData(selectedStore);
  }, [selectedStore, loadStoreData]);

  const activeStore = stores.find((s) => s.store_nbr === selectedStore);

  return (
    <div className="app-shell">
      <Sidebar stores={stores} selectedStore={selectedStore} onSelectStore={setSelectedStore} />

      <main className="main">
        <div className="main-header">
          <div>
            <h1>Demand Forecasting Dashboard</h1>
            <p>HuggingFace Time Series Transformer · next-14-day probabilistic sales forecast</p>
          </div>
          <span className="pill">
            <span className="dot" /> Model online
          </span>
        </div>

        {error && <div className="status-msg err">{error}</div>}

        <OverviewCards overview={overview} />

        <section className="panel">
          <div className="panel-title">
            {activeStore ? `Store #${activeStore.store_nbr} — ${activeStore.city}, ${activeStore.state}` : "Select a store"}
          </div>
          <div className="panel-subtitle">
            Last 120 days of actual sales plus the model's next-14-day forecast with p10–p90 uncertainty band.
          </div>

          {loadingForecast ? (
            <div className="loading-row">
              <span className="spinner" /> Generating forecast...
            </div>
          ) : (
            <SalesChart history={history} forecast={forecast} />
          )}
        </section>

        <div className="grid-2">
          <section className="panel">
            <div className="panel-title">Forecast Detail</div>
            <div className="panel-subtitle">Daily median prediction with lower/upper bounds.</div>
            <ForecastTable forecast={forecast} />
          </section>

          <section className="panel">
            <div className="panel-title">Add New Sales Data</div>
            <div className="panel-subtitle">
              Log today's actual sales for this store to keep the dashboard current.
            </div>
            {selectedStore && (
              <AddDataForm
                storeNbr={selectedStore}
                lastDataDate={overview?.last_date}
                onAdded={() => loadStoreData(selectedStore)}
              />
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
