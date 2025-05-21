import { createApp } from "vue";
import "./assets/main.css";
import App from "./App.vue";
import ui from "@nuxt/ui/vue-plugin";
import { createRouter, createWebHistory } from "vue-router";
import HomeView from "./views/HomeView.vue";
import HistoryView from "./views/HistoryView.vue";

const app = createApp(App);

const routes = [
  { path: "/", component: HomeView },
  { path: "/history", component: HistoryView },
];

const router = createRouter({
  routes,
  history: createWebHistory(),
});

app.use(router);
app.use(ui);

app.mount("#app");
