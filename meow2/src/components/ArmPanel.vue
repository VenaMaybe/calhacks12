<script setup lang="ts">
import { onMounted, onBeforeUnmount, reactive, ref, watch, nextTick } from 'vue'
import type { Marker } from '@/types'

interface Props {
	imageUrl: string
	markers: Marker[]
}
const props = defineProps<Props>()

const boxEl = ref<HTMLElement | null>(null)
const imgEl = ref<HTMLImageElement | null>(null)

// Measured image rect (size) and its offset inside the box
const imgRectInBox = reactive({ w: 0, h: 0, x0: 0, y0: 0 })

function measure() {
	if (!boxEl.value || !imgEl.value) return
	const boxR = boxEl.value.getBoundingClientRect()
	const imgR = imgEl.value.getBoundingClientRect()
	imgRectInBox.w = imgR.width
	imgRectInBox.h = imgR.height
	imgRectInBox.x0 = imgR.left - boxR.left
	imgRectInBox.y0 = imgR.top - boxR.top
}

let ro: ResizeObserver | null = null

onMounted(() => {
	measure()
	window.addEventListener('resize', measure)
	// Re-measure when the actual image finishes loading
	imgEl.value?.addEventListener('load', measure)
	// Observe size changes of the image (responsive layout)
	if ('ResizeObserver' in window && imgEl.value) {
		ro = new ResizeObserver(() => measure())
		ro.observe(imgEl.value)
	}
})

onBeforeUnmount(() => {
	window.removeEventListener('resize', measure)
	imgEl.value?.removeEventListener('load', measure)
	ro?.disconnect()
})

// In case the src changes dynamically
watch(() => props.imageUrl, async () => {
	await nextTick()
	measure()
})

function styleFor(m: Marker) {
	const left = imgRectInBox.x0 + (m.xPct / 100) * imgRectInBox.w
	const top = imgRectInBox.y0 + (m.yPct / 100) * imgRectInBox.h
	return { left: `${left}px`, top: `${top}px` }
}
</script>

<template>
	<div class="arm-panel">
		<div class="img-box" ref="boxEl">
			<img ref="imgEl" class="arm-img" :src="imageUrl" alt="arm diagram" />
			<div
				v-for="m in props.markers"
				:key="m.id"
				class="marker"
				:class="m.state"
				:style="styleFor(m)"
			>
				<span class="dot" />
				<span class="tag" v-if="m.label">{{ m.label }}</span>
			</div>
		</div>
	</div>
</template>

<style scoped>
.arm-panel {
	display: grid;
	place-items: center;
	width: 100%;
	height: 100%;
	padding: 1rem;
}
.img-box {
	position: relative;
	width: 100%;
	height: 100%;
	border-radius: 12px;
	overflow: hidden;
	background: #ffffff;
}
.arm-img {
	display: block;
	max-width: 75%;
	max-height: 75%;
	margin: auto;
	object-fit: contain;
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
}
.marker {
	position: absolute;
	transform: translate(-50%, -50%);
	display: inline-flex;
	align-items: center;
	gap: .25rem;
	--c: #888;
}
.marker .dot {
	width: 12px;
	height: 12px;
	border-radius: 50%;
	background: var(--c);
	box-shadow: 0 0 12px 2px var(--c);
	animation: ping 1.2s infinite;
}
.marker.off .dot { box-shadow: none; }
.marker .tag {
	padding: .1rem .4rem;
	font-size: .8rem;
	border-radius: 6px;
	background: rgba(255,255,255,.06);
	backdrop-filter: blur(2px);
	color: #121214;
}
.marker.on { --c: #34d399; }
.marker.warn { --c: #f59e0b; }
.marker.off { --c: #6b7280; opacity: .7; }

@keyframes ping {
	0% { box-shadow: 0 0 0 0 var(--c); }
	70% { box-shadow: 0 0 0 8px rgba(0,0,0,0); }
	100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
}
</style>
