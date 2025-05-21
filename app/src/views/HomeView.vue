<script setup lang="ts">
import { useDevicesList, useStorage, useUserMedia } from "@vueuse/core";
import InputFileDrop from "../components/InputFileDrop.vue";
import {
  computed,
  reactive,
  ref,
  shallowRef,
  useTemplateRef,
  watch,
  watchEffect,
} from "vue";

const currentCamera = shallowRef<string>();
const { videoInputs: cameras } = useDevicesList({
  requestPermissions: true,
  onUpdated() {
    if (!cameras.value.find((i) => i.deviceId === currentCamera.value))
      currentCamera.value = cameras.value[0]?.deviceId;
  },
});
const currentCameraObj = computed(() =>
  cameras.value.find(({ deviceId: id }) => id == currentCamera.value),
);

const src = ref<string>("");

const video = useTemplateRef<HTMLVideoElement>("video");
const canvas = useTemplateRef<HTMLCanvasElement>("canvas");
const { stream, enabled } = useUserMedia({
  constraints: reactive({
    video: { deviceId: currentCamera, aspectRatio: 0.75 },
  }),
});

const file = ref<File>();

const results = ref<{
  age: number;
  label: string;
  variety: string;
}>();

const state = useStorage<
  { image: string; age: number; label: string; variety: string }[]
>("history", []);

async function onSubmit() {
  if (file.value) {
    const data = new FormData();
    data.append("file", file.value);

    const response = await fetch("http://10.247.194.173:5001/api/predict", {
      method: "POST",
      body: data,
    });

    const prediction = await response.json();

    results.value = prediction;

    state.value = [...state.value, { ...prediction, image: file.value.name }];
  }
}

async function capture() {
  const track = stream.value?.getVideoTracks()[0];

  if (track && canvas.value) {
    // @ts-ignore
    let imageCapture = new ImageCapture(track);
    const image: ImageBitmap = await imageCapture.grabFrame();
    canvas.value.width = image.width;
    canvas.value.height = image.height;
    canvas.value.getContext("2d")?.drawImage(image, 0, 0);
    canvas.value.toBlob((blob) => {
      if (blob) file.value = new File([blob], `predict_${Date.now()}.jpg`);
      onSubmit();
    });
  }
}

function reset() {
  file.value = undefined;
  results.value = undefined;
}

watch([file], () => {
  if (file.value) {
    src.value = URL.createObjectURL(file.value);
    results.value = undefined;
  }
});

watchEffect(() => {
  // preview on a video element
  if (video.value && stream.value) video.value.srcObject = stream.value;
});
</script>

<template>
  <UContainer
    class="space-y-9 min-h-screen flex items-center flex-col justify-center"
  >
    <h1 class="text-2xl font-bold">Rice Plant Disease Prediction</h1>

    <img v-if="src" :src class="w-72 h-auto rounded-lg" />

    <UCard
      v-else
      class="w-72 h-auto aspect-[3/4] text-muted flex items-center justify-center overflow-hidden"
      :ui="{ body: 'p-0 sm:p-0 h-full w-full' }"
    >
      <div
        v-if="!enabled"
        class="text-center p-3 flex items-center justify-center flex-col h-full gap-3"
      >
        <UButtonGroup>
          <UButton
            icon="i-lucide-camera"
            variant="outline"
            color="neutral"
            @click="enabled = !enabled"
          >
            <span v-if="currentCamera && currentCameraObj">
              {{ currentCameraObj.label }}
            </span>
            <span v-else>Select your camera</span>
          </UButton>
          <UDropdownMenu :items="cameras">
            <UButton
              icon="i-lucide-chevron-down"
              color="neutral"
              variant="outline"
            />
          </UDropdownMenu>
        </UButtonGroup>

        <USeparator label="or" />

        <div>Upload an image to get started</div>
      </div>

      <div v-else class="relative h-full w-full">
        <video ref="video" muted autoplay class="h-full w-full object-cover" />

        <div class="absolute bottom-3 right-3 flex items-center gap-3">
          <UButton color="error" @click="enabled = !enabled">End</UButton>
          <UButton icon="i-lucide-camera" color="neutral" @click="capture">
            Capture
          </UButton>
        </div>
      </div>
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

    <InputFileDrop v-else v-model="file" class="w-full" />

    <div class="flex gap-3">
      <UButton variant="soft" v-if="results" @click="reset"> Return </UButton>
      <UButton v-else :disabled="!file" @click="onSubmit">Predict</UButton>
    </div>

    <canvas ref="canvas" width="150" height="150" class="hidden"></canvas>
  </UContainer>
</template>
