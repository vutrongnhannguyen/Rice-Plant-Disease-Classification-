<script setup lang="ts">
import { useStorage } from "@vueuse/core";

const state = useStorage<
  { image: string; age: number; label: string; variety: string }[]
>("history", []);
</script>

<template>
  <UContainer>
    <div
      v-if="state.length > 0"
      class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6"
    >
      <UCard
        v-for="entry in state"
        class="overflow-hidden space-y-3"
        :ui="{ body: 'p-0 sm:p-0 h-full w-full' }"
      >
        <div>
          <img :src="`http://10.247.194.173:5001/uploads/${entry.image}`" />
        </div>
        <ul class="p-3">
          <li><b>Age</b>: {{ entry.age }}</li>
          <li>
            <b>Label</b>:
            <span :class="{ 'text-green-500': entry.label == 'normal' }">{{
              entry.label
            }}</span>
          </li>
          <li><b>Variety</b>: {{ entry.variety }}</li>
        </ul>
      </UCard>
    </div>
    <div
      v-else
      class="flex items-center justify-center min-h-screen flex-col gap-6"
    >
      <UIcon name="i-lucide-history" class="size-18" />
      <div>Currently, you haven't taken any predictions!</div>
      <UButton to="/">Go take some!</UButton>
    </div>
  </UContainer>
</template>
