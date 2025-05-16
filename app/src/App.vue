<script setup lang="ts">
import InputFileDrop from "./components/InputFileDrop.vue";
import { ref, watch } from "vue";

const src = ref<string>("");

const file = ref<File>();
const results = ref<{
  age: number;
  label: string;
  variety: string;
}>();

async function onSubmit() {
  if (file.value) {
    const data = new FormData();
    data.append("file", file.value);

    const response = await fetch("http://localhost:5000/api/predict", {
      method: "POST",
      body: data,
    });

    const prediction = await response.json();

    results.value = prediction;
  }
}

watch([file], () => {
  if (file.value) {
    src.value = URL.createObjectURL(file.value);
    results.value = undefined;
  }
});
</script>

<template>
  <UApp>
    <UContainer
      class="space-y-9 min-h-screen flex items-center flex-col justify-center"
    >
      <h1 class="text-2xl font-bold">Rice Plant Disease Prediction</h1>

      <img v-if="src" :src class="w-72 h-auto rounded-lg" />

      <UCard
        v-else
        class="w-72 h-auto aspect-[3/4] text-muted flex items-center justify-center"
      >
        Upload an image to get started
      </UCard>

      <UCard v-if="results">
        <ul>
          <li><b>Age</b>: {{ results.age }}</li>
          <li>
            <b>Label</b>:
            <span :class="{ 'text-green-500': results.label == 'normal' }">{{
              results.label
            }}</span>
          </li>
          <li><b>Variety</b>: {{ results.variety }}</li>
        </ul>
      </UCard>

      <InputFileDrop v-model="file" class="w-full" />

      <UButton :disabled="!file" @click="onSubmit">Predict</UButton>
    </UContainer>
  </UApp>
</template>
