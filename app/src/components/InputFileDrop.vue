<script setup lang="ts">
import { ref } from "vue";

const props = defineProps<{
  multiple?: boolean;
}>();

const model = defineModel<File | File[]>();

const emit = defineEmits<{
  change: [File | File[]];
}>();

const input = ref<HTMLInputElement>();
const dragging = ref(false);

function open() {
  input.value?.click();
}

function dragover() {
  dragging.value = true;
}

function dragleave() {
  dragging.value = false;
}

function change(event: Event | DragEvent) {
  const files =
    event instanceof DragEvent
      ? event.dataTransfer?.files
      : (event.target as HTMLInputElement).files;

  // reset on change
  model.value = undefined;

  if (files) {
    for (const file of files) {
      if (props.multiple) {
        if (Array.isArray(model.value)) model.value = model.value.concat(file);
        // uninitiated
        else model.value = [file];
      } else {
        model.value = file;
        // set only the first file
        break;
      }
    }

    dragging.value = false;

    if (model.value) emit("change", model.value);
  }
}

const ui = {
  root: "text-center",
  body: "flex items-center justify-center flex-col gap-3 cursor-pointer hover:dark:bg-gray-800/30 transition-colors hover:bg-gray-100",
};
</script>

<template>
  <input ref="input" type="file" class="hidden" :multiple @change="change" />
  <div
    @drop.prevent="change"
    @dragover.prevent="dragover"
    @dragleave="dragleave"
    @click="open"
  >
    <UCard :ui>
      <UIcon name="i-fluent-arrow-upload-20-filled" class="w-8 h-8" />
      <template v-if="dragging">Release to upload</template>
      <template v-else-if="model">
        {{
          Array.isArray(model)
            ? model.map(({ name }) => name).join(", ")
            : model.name
        }}
      </template>
      <template v-else>
        <span><b>Choose a file</b> or drag to upload</span>
      </template>
    </UCard>
  </div>
</template>
