<script setup>
import { ref, computed } from 'vue'

// State management
const activeDomain = ref('skin') // 'skin' or 'cancer'
const selectedFile = ref(null)
const imagePreview = ref(null)
const isLoading = ref(false)
const errorMessage = ref('')
const result = ref(null)
const showGradCam = ref(false)

// Domain configurations
const domainConfig = {
  skin: {
    name: 'Skin Disease Analysis',
    endpoint: 'http://localhost:8000/predict/skin',
    description:
      'Multi-class classification across dermatological conditions (e.g., HAM10000 classes).',
  },
  cancer: {
    name: 'Skin Cancer Detection',
    endpoint: 'http://localhost:8000/predict/cancer',
    description: 'Binary classification focusing on lesion malignancy (Benign vs. Malignant).',
  },
}

// Switch domain and reset workspace
const switchDomain = (domain) => {
  activeDomain.value = domain
  resetWorkspace()
}

// Handle file selection (drag-and-drop or file browser)
const handleFileDrop = (e) => {
  const file = e.dataTransfer?.files?.[0] || e.target.files?.[0]
  if (!file) return

  errorMessage.value = ''
  result.value = null

  // Validate file type (JPEG / PNG) and size (Max 10MB)
  if (!['image/jpeg', 'image/png'].includes(file.type)) {
    errorMessage.value = 'Invalid file format. Please upload a valid JPEG or PNG image.'
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    errorMessage.value = 'File size exceeds the 10MB limit.'
    return
  }

  selectedFile.value = file
  imagePreview.value = URL.createObjectURL(file)
}

// Trigger backend REST API inference
const runInference = async () => {
  if (!selectedFile.value) return

  isLoading.value = true
  errorMessage.value = ''

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const response = await fetch(domainConfig[activeDomain.value].endpoint, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      if (response.status === 400) throw new Error('Bad Request: Invalid file or payload.')
      if (response.status === 415) throw new Error('Unsupported Media Type.')
      if (response.status === 503) throw new Error('Model service not ready / loading error.')
      throw new Error(`Server error: ${response.statusText}`)
    }

    const data = await response.json()
    result.value = data
  } catch (err) {
    errorMessage.value = err.message || 'Failed to connect to the backend server.'
  } finally {
    isLoading.value = false
  }
}

// Reset workspace
const resetWorkspace = () => {
  selectedFile.value = null
  imagePreview.value = null
  result.value = null
  errorMessage.value = ''
  showGradCam.value = false
}

// Sorted probabilities for dashboard
const sortedProbabilities = computed(() => {
  if (!result.value?.probabilities) return []
  return Object.entries(result.value.probabilities).sort((a, b) => b[1] - a[1])
})
</script>

<template>
  <div class="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
    <!-- Persistent Medical Disclaimer Banner -->
    <header
      class="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2.5 text-center text-xs text-amber-300 font-medium flex items-center justify-center gap-2"
    >
      <svg
        class="w-4 h-4 text-amber-400 shrink-0"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <span
        ><strong>CRITICAL DISCLAIMER:</strong> MEDSCAN is developed solely as an educational,
        academic, and research prototype. Not for clinical diagnosis.</span
      >
    </header>

    <!-- Main Navigation / Title -->
    <nav
      class="border-b border-slate-800 bg-slate-950/50 px-6 py-4 flex items-center justify-between"
    >
      <div class="flex items-center gap-3">
        <div class="bg-cyan-500/10 border border-cyan-500/30 p-2 rounded-xl text-cyan-400">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"
            />
          </svg>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white">
            MEDSCAN
            <span
              class="text-cyan-400 text-xs px-2 py-0.5 bg-cyan-500/10 border border-cyan-500/20 rounded-full ml-2"
              >Dual-Model Diagnostic Suite</span
            >
          </h1>
          <p class="text-xs text-slate-400">AI-Powered Dermatological Inference Workstation</p>
        </div>
      </div>

      <!-- Domain Selector Module -->
      <div class="flex bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
        <button
          @click="switchDomain('skin')"
          :class="
            activeDomain === 'skin'
              ? 'bg-cyan-600 text-white shadow'
              : 'text-slate-400 hover:text-white'
          "
          class="px-4 py-1.5 rounded-lg text-xs font-semibold transition-all"
        >
          Skin Disease
        </button>
        <button
          @click="switchDomain('cancer')"
          :class="
            activeDomain === 'cancer'
              ? 'bg-cyan-600 text-white shadow'
              : 'text-slate-400 hover:text-white'
          "
          class="px-4 py-1.5 rounded-lg text-xs font-semibold transition-all"
        >
          Skin Cancer
        </button>
      </div>
    </nav>

    <!-- Main Content Grid -->
    <main
      class="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start"
    >
      <!-- Left Panel: Upload & Preview -->
      <section
        class="lg:col-span-5 bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur flex flex-col gap-5"
      >
        <div>
          <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Input Source
          </h2>
          <p class="text-xs text-slate-400 mt-0.5">{{ domainConfig[activeDomain].description }}</p>
        </div>

        <!-- Drag-and-Drop Dropzone -->
        <div
          @dragover.prevent
          @drop.prevent="handleFileDrop"
          class="relative border-2 border-dashed border-slate-600 hover:border-cyan-500 rounded-xl p-6 transition-all flex flex-col items-center justify-center text-center cursor-pointer bg-slate-900/40 min-h-[240px]"
        >
          <input
            type="file"
            @change="handleFileDrop"
            accept="image/jpeg, image/png"
            class="absolute inset-0 opacity-0 cursor-pointer"
          />

          <div v-if="!imagePreview" class="flex flex-col items-center gap-3">
            <div class="p-3 bg-slate-800 rounded-full text-cyan-400 border border-slate-700">
              <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="1.5"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div>
              <p class="text-sm font-medium text-slate-200">Click to upload or drag & drop</p>
              <p class="text-xs text-slate-400 mt-1">Supports JPEG, PNG (Max 10MB)</p>
            </div>
          </div>

          <!-- Image Preview -->
          <div v-else class="relative w-full h-full flex flex-col items-center">
            <img
              :src="
                showGradCam && result?.gradcam
                  ? `data:image/jpeg;base64,${result.gradcam}`
                  : imagePreview
              "
              class="max-h-60 rounded-lg object-contain border border-slate-700 shadow-md"
              alt="Uploaded scan preview"
            />
            <span class="text-xs text-slate-400 mt-2 truncate max-w-[220px]">{{
              selectedFile?.name
            }}</span>
          </div>
        </div>

        <!-- Error Alert -->
        <div
          v-if="errorMessage"
          class="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs"
        >
          {{ errorMessage }}
        </div>

        <!-- Action Controls -->
        <div class="flex gap-3">
          <button
            @click="runInference"
            :disabled="!selectedFile || isLoading"
            class="flex-1 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold py-2.5 px-4 rounded-xl text-sm transition-all shadow-lg flex items-center justify-center gap-2"
          >
            <svg
              v-if="isLoading"
              class="animate-spin w-4 h-4 text-white"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                class="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                stroke-width="4"
              ></circle>
              <path
                class="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            <span>{{ isLoading ? 'Running Inference...' : 'Analyze Image' }}</span>
          </button>
          <button
            @click="resetWorkspace"
            class="bg-slate-700 hover:bg-slate-600 text-slate-300 font-medium py-2.5 px-4 rounded-xl text-sm transition-all"
          >
            Reset
          </button>
        </div>
      </section>

      <!-- Right Panel: Results & Probability Dashboard -->
      <section
        class="lg:col-span-7 bg-slate-800/40 border border-slate-700/60 rounded-2xl p-6 backdrop-blur flex flex-col gap-6 min-h-[460px]"
      >
        <div class="flex items-center justify-between border-b border-slate-700/60 pb-4">
          <div>
            <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Diagnostic Dashboard
            </h2>
            <p class="text-xs text-slate-400 mt-0.5">
              Classification predictions & Grad-CAM explainability
            </p>
          </div>
          <!-- Grad-CAM Toggle -->
          <div
            v-if="result?.gradcam"
            class="flex items-center gap-2 bg-slate-900/60 px-3 py-1.5 rounded-xl border border-slate-700"
          >
            <span class="text-xs text-slate-300">Grad-CAM Overlay</span>
            <input
              type="checkbox"
              v-model="showGradCam"
              class="accent-cyan-500 w-4 h-4 cursor-pointer"
            />
          </div>
        </div>

        <!-- Initial Empty State -->
        <div
          v-if="!result && !isLoading"
          class="flex-1 flex flex-col items-center justify-center text-center text-slate-500 py-12"
        >
          <svg
            class="w-12 h-12 mb-3 stroke-slate-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
              d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p class="text-sm">Awaiting image upload and analysis...</p>
        </div>

        <!-- Loading State -->
        <div v-if="isLoading" class="flex-1 flex flex-col items-center justify-center py-12">
          <div
            class="w-10 h-10 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mb-4"
          ></div>
          <p class="text-sm text-cyan-400 font-medium">
            Executing deep learning inference & generating Grad-CAM...
          </p>
        </div>

        <!-- Prediction Results View -->
        <div v-if="result && !isLoading" class="flex flex-col gap-6">
          <!-- Top Classification Card -->
          <div
            class="bg-gradient-to-r from-cyan-950/40 to-slate-900/60 border border-cyan-500/30 p-5 rounded-xl flex items-center justify-between"
          >
            <div>
              <span class="text-xs font-semibold text-cyan-400 uppercase tracking-widest"
                >Top Prediction</span
              >
              <h3 class="text-xl font-bold text-white mt-1">{{ result.class }}</h3>
            </div>
            <div class="text-right">
              <span class="text-xs text-slate-400 block">Confidence Score</span>
              <span class="text-2xl font-black text-cyan-400"
                >{{ (result.confidence * 100).toFixed(2) }}%</span
              >
            </div>
          </div>

          <!-- Probability Distribution Breakdown -->
          <div>
            <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Probability Distribution Across Classes
            </h4>
            <div class="flex flex-col gap-2.5 max-h-[220px] overflow-y-auto pr-1">
              <div
                v-for="[className, prob] in sortedProbabilities"
                :key="className"
                class="bg-slate-900/40 p-3 rounded-xl border border-slate-700/60"
              >
                <div class="flex justify-between text-xs mb-1.5 font-medium">
                  <span class="text-slate-200">{{ className }}</span>
                  <span class="text-cyan-400">{{ (prob * 100).toFixed(1) }}%</span>
                </div>
                <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div
                    class="bg-cyan-500 h-full rounded-full transition-all duration-500"
                    :style="{ width: `${prob * 100}%` }"
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>

    <!-- Footer -->
    <footer
      class="border-t border-slate-800 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500"
    >
      MEDSCAN Internship Project — Vue.js 3 & Tailwind CSS Frontend Workstation
    </footer>
  </div>
</template>
