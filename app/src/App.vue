<script setup lang="ts">
import type { NavigationMenuItem } from "@nuxt/ui";
import { ref } from "vue";
import { RouterView } from "vue-router";

const items = ref<NavigationMenuItem[]>([
  {
    label: "Home",
    icon: "i-lucide-house",
    to: "/",
  },
  {
    label: "History",
    icon: "i-lucide-history",
    to: "/history",
  },
]);
</script>

<template>
  <UApp>
    <UNavigationMenu :items="items" class="w-full justify-center" />

    <RouterView />
  </UApp>
</template>
