<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { HandOption, Recommendation } from '@/types'

interface Props {
	items: HandOption[]
	recommendation?: Recommendation | null
}
const props = defineProps<Props>()
const idx = ref(0)

watch(
	() => props.recommendation?.handId,
	(id) => {
		if (!id) return
		const i = props.items.findIndex(h => h.id === id)
		if (i >= 0) idx.value = i
	}
)

function next() {
	if (!props.items.length) return
	idx.value = (idx.value + 1) % props.items.length
}
function prev() {
	if (!props.items.length) return
	idx.value = (idx.value - 1 + props.items.length) % props.items.length
}

const current = computed(() => props.items[idx.value])
</script>

<template>
	<div class="carousel" v-if="items.length">
		<div class="stage">
			<img :src="current.imageUrl" :alt="current.label" />
		</div>
		<div class="meta">
			<div class="title">
				{{ current.label }}
				<span v-if="current.score !== undefined" class="score">({{ (current.score * 100).toFixed(0) }}%)</span>
			</div>
			<div class="controls">
				<button @click="prev" aria-label="previous">‹</button>
				<button @click="next" aria-label="next">›</button>
			</div>
		</div>
		<div class="thumbs">
			<button
				v-for="(h, i) in items"
				:key="h.id"
				:class="['thumb', { active: i === idx }]"
				@click="idx = i"
				:title="h.label"
			>
				<img :src="h.imageUrl" :alt="h.label" />
			</button>
		</div>
	</div>
</template>

<style scoped>
.carousel {
	display: grid;
	grid-template-rows: 1fr auto auto;
	gap: .75rem;
	height: 100%;
	padding: 1rem;
}
.stage {
	display: grid;
	place-items: center;
	min-height: 200px;
	border-radius: 12px;
	background: #ffffff;
	color: #121214;
}
.stage img {
	max-width: 75%;
	max-height: 75%;
	object-fit: contain;
}
.meta { display: flex; justify-content: space-between; align-items: center; }
.title { font-weight: 600; color: #121214; }
.score { margin-left: .35rem; color: #a7a7ff; }
.controls button {
	font-size: 1.2rem;
	padding: .25rem .6rem;
	border-radius: 8px;
	border: 1px solid #a7a7ff;
	background: #e5e7eb;
	color: #121214;
}
.thumbs {
	display: grid;
	grid-auto-flow: column;
	gap: .5rem;
	overflow-x: auto;
	padding-bottom: .25rem;
}
.thumb {
	border: 1px solid transparent;
	border-radius: 8px;
	background: transparent;
	padding: .25rem;
	color: #000000;
}
.thumb.active {
	background: #ccccff;
	border-color: #a7a7ff;
}
.thumb img {
	width: 56px;
	height: 56px;
	object-fit: contain;
}
</style>
