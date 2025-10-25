import { defineStore } from 'pinia'
import type { Marker, HandOption, Recommendation } from '@/types'

export const useArmStore = defineStore('arm', {
	state: () => ({
		markers: [] as Marker[],
		hands: [] as HandOption[],
		recommendation: null as Recommendation | null,
		loading: false,
		error: '' as string | null
	}),
	actions: {
		async bootstrap() {
			this.loading = true
			try {
				await Promise.all([this.fetchMarkers(), this.fetchHands(), this.fetchRecommendation()])
			} catch (e: any) {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
				this.error = e?.message ?? 'Failed to load'
			} finally {
				this.loading = false
			}
		},
		async fetchMarkers() {
			this.markers = [
				{ id: 'elbow', xPct: 35, yPct: 55, state: 'on', label: 'Elbow' },
				{ id: 'wrist', xPct: 62, yPct: 42, state: 'warn', label: 'Wrist' },
				{ id: 'fingers', xPct: 78, yPct: 22, state: 'off', label: 'Fingers' }
			]
		},
		async fetchHands() {
			this.hands = [
				{ id: 'hand-a', label: 'Open Hand', imageUrl: '/hands/open.png', score: 0.41 },
				{ id: 'hand-b', label: 'Grip', imageUrl: '/hands/grip.png', score: 0.83 },
				{ id: 'hand-c', label: 'Pinch', imageUrl: '/hands/pinch.png', score: 0.37 }
			]
		},
		async fetchRecommendation() {
			this.recommendation = { handId: 'hand-b', confidence: 0.83 }
		},
		selectHand(id: string) {
			this.recommendation = { handId: id }
		}
	}
})
