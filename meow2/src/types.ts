export type MarkerState = 'off' | 'warn' | 'on'

export interface Marker {
	id: string
	// percentages [0..100] relative to the image box
	xPct: number
	yPct: number
	state: MarkerState
	label?: string
}

export interface HandOption {
	id: string
	label: string
	imageUrl: string
	score?: number
}

export interface Recommendation {
	handId: string
	confidence?: number
}