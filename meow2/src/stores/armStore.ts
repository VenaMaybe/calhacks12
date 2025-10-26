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
				{ id: 'flexor', xPct: 43, yPct: 80, state: 'on', label: 'Flexor Capri Radialis' },
				{ id: 'extensor', xPct: 62, yPct: 42, state: 'warn', label: 'Extensor Digitorum Communis' },
			]
		},
		async fetchHands() {
			this.hands = [
				{ id: 'hand-rock', label: 'Rock', imageUrl: '/hands/rock.png', score: 0.37 },
				{ id: 'hand-paper', label: 'Paper', imageUrl: '/hands/paper.png', score: 0.37 },
				{ id: 'hand-scissors', label: 'Scissors', imageUrl: '/hands/scissors.png', score: 0.37 }
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
