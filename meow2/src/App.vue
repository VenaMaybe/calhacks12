<script setup lang="ts">
import ArmPanel from '@/components/ArmPanel.vue'
import HandCarousel from '@/components/HandCarousel.vue'
import { useArmStore } from '@/stores/armStore'
import { onMounted } from 'vue'

const store = useArmStore()

onMounted(() => {
	store.bootstrap()
	// If you want live updates, poll or open a WebSocket here.
	// Example poll:
	// setInterval(() => store.fetchMarkers().then(store.fetchRecommendation), 1500)
})
</script>

<template>
	<div class="layout">
		<section class="left">
			<ArmPanel imageUrl="/arm.png" :markers="store.markers" />
		</section>
		<section class="right">
			<HandCarousel :items="store.hands" :recommendation="store.recommendation" />
		</section>
	</div>
</template>

<style scoped>
.layout {
	display: grid;
	grid-template-columns: 1.2fr 1fr;
	gap: 1rem;
	height: 100dvh;
	padding: 1rem;
	box-sizing: border-box;
	background: #0a0a0b;
}
.left, .right {
	background: #0f0f12;
	border: 1px solid #1f1f24;
	border-radius: 16px;
	overflow: hidden;
}
@media (max-width: 900px) {
	.layout { grid-template-columns: 1fr; height: auto; }
	.left, .right { min-height: 50dvh; }
}
</style>