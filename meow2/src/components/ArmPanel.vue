<script setup lang="ts">
import { ref, watchEffect } from 'vue'
import type { Marker } from '@/types'

interface Props {
	imageUrl: string
	markers: Marker[]
}
const props = defineProps<Props>()

// Natural (intrinsic) image size for precise viewBox
const natW = ref<number>(0)
const natH = ref<number>(0)

async function loadNaturalSize(url: string) {
	return new Promise<{ w: number; h: number }>((resolve, reject) => {
		const img = new Image()
		img.onload = () => resolve({ w: img.naturalWidth || img.width, h: img.naturalHeight || img.height })
		img.onerror = reject
		img.src = url
	})
}

watchEffect(async () => {
	if (!props.imageUrl) return
	try {
		const { w, h } = await loadNaturalSize(props.imageUrl)
		natW.value = w || 1000
		natH.value = h || 1000
	} catch {
		// sensible fallback
		natW.value = 1000
		natH.value = 562
	}
})

function toXY(m: Marker) {
	return {
		x: (m.xPct / 100) * natW.value,
		y: (m.yPct / 100) * natH.value
	}
}
</script>

<template>
	<div class="arm-panel">
		<!-- The wrapper keeps a nice box; SVG handles aspect + letterboxing internally -->
		<div
			class="svg-wrap"
			:style="{ aspectRatio: natW && natH ? `${natW}/${natH}` : '16/9' }"
		>
			<div class="svg-inset">
				<svg
					class="arm-svg"
					:viewBox="`0 0 ${natW} ${natH}`"
					preserveAspectRatio="xMidYMid meet"
				>
					<image :href="props.imageUrl" x="0" y="0" :width="natW" :height="natH" pointer-events="none" />
					<g v-for="m in props.markers" :key="m.id" class="marker" :class="m.state" :transform="`translate(${toXY(m).x}, ${toXY(m).y})`">
						<circle class="dot" r="6" />
						<text class="tag" x="10" y="4">{{ m.label }}</text>
					</g>
				</svg>
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

.svg-wrap {
	position: relative;
	width: 100%;
	height: 100%;
	border-radius: 12px;
	overflow: hidden;
	background: #ffffff;
	--inset: 0.9; /* change this to .8, .6, etc. */
}

.svg-inset {
	position: absolute;
	inset: 0;
	display: grid;
	place-items: center;
}

.arm-svg {
	/* 75% inset inside the white box */
	width: calc(var(--inset) * 100%);
	height: calc(var(--inset) * 100%);
	display: block;
}

/* Marker styling (SVG-friendly) */
.marker { --c: #888; }
.marker.on { --c: #69db7d; }
.marker.warn { --c: #f59e0b; }
.marker.off { --c: #6b7280; opacity: .8; }

.dot {
	fill: var(--c);
	filter: drop-shadow(0 0 6px var(--c));
	animation: ping 1.2s infinite;
}
.marker.off .dot { filter: none; animation: none; }

.tag {
	font-size: .8rem;
	fill: #121214;
	paint-order: stroke;
	stroke: rgba(255,255,255,.7);
	stroke-width: 6px;
	stroke-linejoin: round;
}

@keyframes ping {
	0% { filter: drop-shadow(0 0 0 var(--c)); }
	70% { filter: drop-shadow(0 0 8px rgba(0,0,0,0)); }
	100% { filter: drop-shadow(0 0 0 rgba(0,0,0,0)); }
}
</style>
