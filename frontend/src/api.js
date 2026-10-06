import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({ baseURL: BASE_URL, timeout: 30000 });

export const getStores = () => client.get("/api/stores").then((r) => r.data);
export const getOverview = () => client.get("/api/overview").then((r) => r.data);
export const getHistory = (storeNbr, days = 120) =>
  client.get(`/api/history/${storeNbr}`, { params: { days } }).then((r) => r.data);
export const getForecast = (storeNbr) =>
  client.get(`/api/forecast/${storeNbr}`).then((r) => r.data);
export const addDataRecord = (record) =>
  client.post("/api/data", record).then((r) => r.data);
export const getDataLog = () => client.get("/api/data/log").then((r) => r.data);

export default client;
