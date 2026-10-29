import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import L from 'leaflet'
import { CircleMarker, MapContainer, Polygon, Popup, ScaleControl, TileLayer, Tooltip as MapTooltip, ZoomControl, useMap, useMapEvents } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const API = '/api'

const primaryNavItems = [
  'Dashboard',
  'Map',
  'Micro-Watershed Ranking',
  'Water Conservation Recommendations',
  'Predict New Micro-Watershed',
]

const moreNavItems = [
  'Predictive Analysis',
  'Conservation Structure Suitability',
  'Scenario Simulator',
  'AI Explainability',
  'Model Performance',
  'Future Prediction',
  'Reports & Batch',
  'Methodology',
  'Watershed Insights',
]

const conservationStructures = [
  'Check Dam',
  'Farm Pond',
  'Percolation Pond',
  'Recharge Structure / Recharge Pit',
  'Contour Bund',
  'Vegetative / Soil Conservation',
]

const corePredictionInputs = [
  ['Month', 'Month'],
  ['Year', 'Year'],
  ['Latitude', 'Latitude'],
  ['Longitude', 'Longitude'],
  ['Mean_Elevation_m', 'Mean elevation (m)'],
  ['Rainfall_mm', 'Rainfall (mm)'],
  ['Groundwater_Depth_m', 'Groundwater depth (m)'],
  ['Mean_Slope_deg', 'Mean slope (°)'],
  ['Soil_Moisture_m3_m3', 'Soil moisture'],
  ['NDVI', 'Vegetation index (NDVI)'],
]

const initialForm = {
  Month: 7,
  Year: 2026,
  Latitude: 11.54505,
  Longitude: 79.31001,
  Mean_Elevation_m: 110.78,
  Rainfall_mm: 55,
  Tmax_C: 30,
  Tmin_C: 22,
  Mean_Temperature_C: 26,
  Relative_Humidity_pct: 68,
  PET_mm: 120,
  AET_mm: 75,
  Soil_Moisture_m3_m3: 0.22,
  Runoff_mm: 14,
  Streamflow_m3_s: 0.8,
  Groundwater_Depth_m: 6.5,
  Sand_pct: 35,
  Clay_pct: 32,
  Silt_pct: 33,
  Soil_Organic_Carbon_pct: 0.82,
  Soil_pH: 7.2,
  Available_Water_Capacity: 0.18,
  Infiltration_Rate_mm_hr: 2.5,
  NDVI: 0.42,
  NDWI: 0.18,
  Agriculture_pct: 48,
  Forest_pct: 18,
  Builtup_pct: 12,
  Waterbody_pct: 2,
  Barren_Land_pct: 20,
  Lineament_Density_km_per_km2: 1.7,
  Geological_Permeability_Index: 0.65,
  Population_Density_per_km2: 1150,
  Mean_Slope_deg: 8.5,
  Max_Slope_deg: 21,
  Drainage_Density_km_per_km2: 2.8,
  Stream_Frequency_per_km2: 1.3,
  Stream_Order: 3,
}

function numberOr(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback
}

const MAP_AREA_COLORS = { High: '#dc2626', Medium: '#facc15', Low: '#16a34a' }
const MAP_AREA_BANDS = [
  { key: 'Low', label: 'Low', color: MAP_AREA_COLORS.Low },
  { key: 'Medium', label: 'Middle', color: MAP_AREA_COLORS.Medium },
  { key: 'High', label: 'High', color: MAP_AREA_COLORS.High },
]

function mapLayerMetricValue(point, layer, structure) {
  if (layer === 'WCHI_Priority') return null

  if (layer === 'Structure_Suitability') {
    const assessment = (point.Structure_Suitability || []).find((item) => item.Structure === structure)
    return Number.isFinite(Number(assessment?.['Suitability Score']))
      ? Number(assessment['Suitability Score'])
      : null
  }

  if (layer === 'Conservation_Recommendation') {
    const recommendation = point.Recommended_Conservation_Structure
    const assessment = (point.Structure_Suitability || []).find((item) => item.Structure === recommendation)
    return Number.isFinite(Number(assessment?.['Suitability Score']))
      ? Number(assessment['Suitability Score'])
      : null
  }

  const value = Number(point[layer])
  return Number.isFinite(value) ? value : null
}

function mapAreaBand(point, layer, structure, minimum, maximum, lowerBreak = 34, upperBreak = 67) {
  if (layer === 'WCHI_Priority') {
    const priority = point.WCHI_Priority || point.Priority
    return ['High', 'Medium', 'Low'].includes(priority) ? priority : null
  }

  const value = mapLayerMetricValue(point, layer, structure)
  if (value == null) return null

  if (layer === 'Structure_Suitability' || layer === 'Conservation_Recommendation') {
    return value >= 70 ? 'High' : value >= 45 ? 'Medium' : 'Low'
  }

  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return null
  const position = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum)
  return position * 100 >= upperBreak ? 'High' : position * 100 >= lowerBreak ? 'Medium' : 'Low'
}

function buildApproximatePriorityRegions(points) {
  const validPoints = points
    .map((point) => ({
      point,
      latitude: Number(point.Latitude),
      longitude: Number(point.Longitude),
    }))
    .filter(({ latitude, longitude }) => (
      Number.isFinite(latitude) && Number.isFinite(longitude)
      && latitude >= -90 && latitude <= 90
      && longitude >= -180 && longitude <= 180
    ))

  if (!validPoints.length) return []

  const longitudeScale = Math.cos(
    validPoints.reduce((sum, item) => sum + item.latitude, 0) / validPoints.length * Math.PI / 180,
  )
  const sites = validPoints.map((item) => ({
    ...item,
    x: item.longitude * longitudeScale,
    y: item.latitude,
    priority: item.point.WCHI_Priority || item.point.Priority,
  }))
  const minimumX = Math.min(...sites.map((site) => site.x))
  const maximumX = Math.max(...sites.map((site) => site.x))
  const minimumY = Math.min(...sites.map((site) => site.y))
  const maximumY = Math.max(...sites.map((site) => site.y))
  const padding = Math.max(maximumX - minimumX, maximumY - minimumY) * 0.05

  return sites.map((site, siteIndex) => {
    let polygon = [
      [minimumX - padding, minimumY - padding],
      [maximumX + padding, minimumY - padding],
      [maximumX + padding, maximumY + padding],
      [minimumX - padding, maximumY + padding],
    ]

    sites.forEach((other, otherIndex) => {
      if (otherIndex === siteIndex || !polygon.length) return

      const a = 2 * (other.x - site.x)
      const b = 2 * (other.y - site.y)
      const c = other.x ** 2 + other.y ** 2 - site.x ** 2 - site.y ** 2
      const clipped = []

      for (let index = 0; index < polygon.length; index += 1) {
        const start = polygon[index]
        const end = polygon[(index + 1) % polygon.length]
        const startDistance = a * start[0] + b * start[1] - c
        const endDistance = a * end[0] + b * end[1] - c
        const startInside = startDistance <= 1e-10
        const endInside = endDistance <= 1e-10

        if (startInside) clipped.push(start)
        if (startInside !== endInside) {
          const ratio = startDistance / (startDistance - endDistance)
          clipped.push([
            start[0] + ratio * (end[0] - start[0]),
            start[1] + ratio * (end[1] - start[1]),
          ])
        }
      }

      polygon = clipped
    })

    return {
      id: site.point.Micro_Watershed_ID,
      priority: site.priority,
      positions: polygon.map(([x, y]) => [y, x / longitudeScale]),
    }
  }).filter((region) => region.positions.length >= 3)
}

function App() {
  const [activePage, setActivePage] = useState('Dashboard')
  const [dashboard, setDashboard] = useState({})
  const [metrics, setMetrics] = useState([])
  const [predictions, setPredictions] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [rankingData, setRankingData] = useState([])
  const [monthlyAnalytics, setMonthlyAnalytics] = useState([])
  const [modelExplainability, setModelExplainability] = useState(null)
  const [siteExplanation, setSiteExplanation] = useState(null)
  const [futureProjection, setFutureProjection] = useState(null)
  const [microWatersheds, setMicroWatersheds] = useState([])
  const [mapData, setMapData] = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [watershedQuery, setWatershedQuery] = useState('')
  const [searchError, setSearchError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [formDefaultsSource, setFormDefaultsSource] = useState('')
  const [formValues, setFormValues] = useState(initialForm)
  const [predictionResult, setPredictionResult] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [predictionLoading, setPredictionLoading] = useState(false)
  const predictionResultRef = useRef(null)
  const predictionDefaultsLoaded = useRef(false)
  const predictionFormTouched = useRef(false)
  const [rankingSearch, setRankingSearch] = useState('')
  const [rankingPriority, setRankingPriority] = useState('All')
  const [rankingSort, setRankingSort] = useState('WCHI_Score')
  const [mapLayer, setMapLayer] = useState('WCHI_Priority')
  const [mapBasemap, setMapBasemap] = useState('dem')
  const [mapView, setMapView] = useState(null)
  const [mapFocusRequest, setMapFocusRequest] = useState(0)
  const [mapSearchQuery, setMapSearchQuery] = useState('')
  const [mapSearchError, setMapSearchError] = useState('')
  const [mapTileError, setMapTileError] = useState(false)
  const mapFrameRef = useRef(null)
  const [showMapMarkers, setShowMapMarkers] = useState(true)
  const [highlightedMapBands, setHighlightedMapBands] = useState({ High: true, Medium: true, Low: true })
  const [mapBandBreaks, setMapBandBreaks] = useState({ lower: 34, upper: 67 })
  const [selectedSuitabilityStructure, setSelectedSuitabilityStructure] = useState('')
  const [scenarioComponents, setScenarioComponents] = useState(null)
  const [scenarioResult, setScenarioResult] = useState(null)
  const [scenarioError, setScenarioError] = useState('')
  const [scenarioLoading, setScenarioLoading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [uploadLoading, setUploadLoading] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [mapPriorityFilter, setMapPriorityFilter] = useState('All')
  const [mapStructureFilter, setMapStructureFilter] = useState('All')

  const loadData = useCallback(async () => {
    try {
      const [dashRes, metricsRes, predRes, recRes, rankingRes, explainRes, futureRes, defaultsRes, monthlyRes, microRes, mapRes] = await Promise.all([
        fetch(`${API}/dashboard`),
        fetch(`${API}/model-metrics`),
        fetch(`${API}/predictions`),
        fetch(`${API}/recommendations`),
        fetch(`${API}/rankings`),
        fetch(`${API}/model-explainability`),
        fetch(`${API}/future-projection`),
        fetch(`${API}/prediction-defaults`),
        fetch(`${API}/analytics/monthly`),
        fetch(`${API}/micro-watersheds`),
        fetch(`${API}/map`),
      ])

      const responses = [dashRes, metricsRes, predRes, recRes, rankingRes, explainRes, futureRes, defaultsRes, monthlyRes, microRes, mapRes]
      if (responses.some((response) => !response.ok)) {
        throw new Error('One or more backend data services are unavailable.')
      }

      const dashboardPayload = await dashRes.json()
      const metricsPayload = await metricsRes.json()
      const predictionsPayload = await predRes.json()
      const recommendationsPayload = await recRes.json()
      const rankingPayload = await rankingRes.json()
      const explainPayload = await explainRes.json()
      const futurePayload = await futureRes.json()
      const defaultsPayload = await defaultsRes.json()
      const monthlyPayload = await monthlyRes.json()
      const microWatershedsPayload = await microRes.json()
      const mapPayload = await mapRes.json()

      setDashboard(dashboardPayload)
      setMetrics(metricsPayload)
      setPredictions(predictionsPayload)
      setRecommendations(recommendationsPayload)
      setRankingData(rankingPayload)
      setModelExplainability(explainPayload)
      setFutureProjection(futurePayload)
      setMonthlyAnalytics(monthlyPayload)
      setMicroWatersheds(microWatershedsPayload)
      setMapData(mapPayload)
      setFormDefaultsSource(defaultsPayload.source || '')
      if (!predictionDefaultsLoaded.current && !predictionFormTouched.current) {
        setFormValues((currentValues) => ({ ...currentValues, ...defaultsPayload.values }))
        predictionDefaultsLoaded.current = true
      }
      setSelectedId((currentId) => (
        microWatershedsPayload.some((row) => String(row.Micro_Watershed_ID) === String(currentId))
          ? currentId
          : microWatershedsPayload[0]?.Micro_Watershed_ID || ''
      ))
      setLastUpdated(new Date())
    } catch (error) {
      setLoadError(error.message)
      console.error('Failed to load data from API', error)
    } finally {
      setLoading(false)
    }
  }, [])

  const refreshData = async () => {
    setLoading(true)
    setLoadError('')
    await loadData()
  }

  useEffect(() => {
    Promise.resolve().then(loadData)
  }, [loadData])

  useEffect(() => {
    if (!selectedId) return
    fetch(`${API}/explanations/${encodeURIComponent(selectedId)}`)
      .then(async (response) => {
        const payload = await response.json()
        if (!response.ok) throw new Error(payload.error || 'Could not load watershed explanation.')
        setSiteExplanation(payload)
      })
      .catch((error) => console.error('Failed to load watershed explanation', error))
  }, [selectedId])

  const selectedWatershed = useMemo(
    () => microWatersheds.find((w) => String(w.Micro_Watershed_ID) === String(selectedId)) || microWatersheds[0],
    [microWatersheds, selectedId],
  )

  const watershedOptions = useMemo(() => {
    const seen = new Set()
    return microWatersheds.filter((row) => {
      const id = String(row.Micro_Watershed_ID)
      if (seen.has(id)) return false
      seen.add(id)
      return true
    })
  }, [microWatersheds])

  const predictedRows = useMemo(
    () => predictions.filter((row) => row.Predicted_WCHI_Score !== null && row.Predicted_WCHI_Score !== undefined),
    [predictions],
  )

  const predictedWatersheds = useMemo(() => {
    const grouped = new Map()
    predictedRows.forEach((row) => {
      const id = String(row.Micro_Watershed_ID)
      const group = grouped.get(id) || {
        Micro_Watershed_ID: id,
        count: 0,
        rainfall: 0,
        recharge: 0,
        wchi: 0,
        priorities: {},
      }
      group.count += 1
      group.rainfall += numberOr(row.Predicted_Rainfall_Score)
      group.recharge += numberOr(row.Predicted_Recharge_Score)
      group.wchi += numberOr(row.Predicted_WCHI_Score)
      const priority = row.Predicted_WCHI_Priority || 'Unknown'
      group.priorities[priority] = (group.priorities[priority] || 0) + 1
      grouped.set(id, group)
    })
    return Array.from(grouped.values()).map((group) => ({
      Micro_Watershed_ID: group.Micro_Watershed_ID,
      Predicted_Rainfall_Score: group.rainfall / group.count,
      Predicted_Recharge_Score: group.recharge / group.count,
      Predicted_WCHI_Score: group.wchi / group.count,
      Predicted_WCHI_Priority: Object.entries(group.priorities).sort(([, first], [, second]) => second - first)[0]?.[0] || 'Unknown',
    }))
  }, [predictedRows])

  const priorityDistribution = useMemo(() => {
    const counts = { High: 0, Medium: 0, Low: 0 }
    predictedWatersheds.forEach((row) => {
      const priority = row.Predicted_WCHI_Priority
      if (priority in counts) counts[priority] += 1
    })
    return Object.entries(counts).map(([name, value]) => ({ name, value }))
  }, [predictedWatersheds])

  const wchiChart = useMemo(
    () => predictedWatersheds.map((row) => ({ name: row.Micro_Watershed_ID, WCHI: numberOr(row.Predicted_WCHI_Score) })),
    [predictedWatersheds],
  )

  const scatterData = useMemo(
    () =>
      predictions.map((row) => ({
        x: numberOr(row.Predicted_WCHI_Score),
        y: numberOr(row.Actual_WCHI_Score),
        z: 10,
        name: row.Micro_Watershed_ID,
      })),
    [predictions],
  )

  const topWatersheds = useMemo(
    () => [...microWatersheds].sort((a, b) => numberOr(b.WCHI_Score) - numberOr(a.WCHI_Score)).slice(0, 8),
    [microWatersheds],
  )

  const structureSummary = useMemo(() => {
    const counts = recommendations.reduce((summary, row) => {
      const structure = row['Recommended Structure'] || row.Recommended_Conservation_Structure || 'Unspecified'
      summary[structure] = (summary[structure] || 0) + 1
      return summary
    }, {})
    return Object.entries(counts).sort(([, first], [, second]) => second - first)
  }, [recommendations])

  const recommendationLookup = useMemo(
    () => new Map(recommendations.map((row) => [String(row.Micro_Watershed_ID), row])),
    [recommendations],
  )

  const averageScores = useMemo(() => {
    const scoreKeys = ['Rainfall_Score', 'Recharge_Score', 'Runoff_Score', 'Groundwater_Score']
    return scoreKeys.map((key) => {
      const values = microWatersheds.map((row) => numberOr(row[key])).filter((value) => Number.isFinite(value))
      const average = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
      return { name: key.replace('_Score', ''), value: average }
    })
  }, [microWatersheds])

  const priorityColors = { High: '#ef4444', Medium: '#f59e0b', Low: '#10b981' }
  const predictionDataReady = Boolean(dashboard.prediction_available && predictedWatersheds.length)

  const mapStructures = useMemo(() => {
    const values = mapData.map((point) => point.Recommended_Conservation_Structure).filter(Boolean)
    return ['All', ...new Set(values)]
  }, [mapData])

  const activeSuitabilityStructure = selectedSuitabilityStructure || conservationStructures[0]

  const elevationRange = useMemo(() => {
    const elevations = mapData.map((point) => Number(point.Mean_Elevation_m)).filter(Number.isFinite)
    return elevations.length
      ? { minimum: Math.min(...elevations), maximum: Math.max(...elevations) }
      : null
  }, [mapData])

  const mapMetricRange = useMemo(() => {
    if (mapLayer === 'WCHI_Priority') return null
    const values = mapData
      .map((point) => mapLayerMetricValue(point, mapLayer, activeSuitabilityStructure))
      .filter((value) => value != null)
    return values.length
      ? { minimum: Math.min(...values), maximum: Math.max(...values) }
      : null
  }, [mapData, mapLayer, activeSuitabilityStructure])

  const getMapPointBand = useCallback(
    (point) => mapAreaBand(
      point,
      mapLayer,
      activeSuitabilityStructure,
      mapMetricRange?.minimum,
      mapMetricRange?.maximum,
      mapBandBreaks.lower,
      mapBandBreaks.upper,
    ),
    [mapLayer, activeSuitabilityStructure, mapMetricRange, mapBandBreaks],
  )

  const candidateMapData = useMemo(
    () => mapData.filter((point) => {
      const priority = point.WCHI_Priority || point.Priority || 'Low'
      const structure = point.Recommended_Conservation_Structure || 'Unspecified'
      return (mapPriorityFilter === 'All' || priority === mapPriorityFilter)
        && (mapStructureFilter === 'All' || structure === mapStructureFilter)
    }),
    [mapData, mapPriorityFilter, mapStructureFilter],
  )

  const mapBandCounts = useMemo(() => {
    const counts = { High: 0, Medium: 0, Low: 0 }
    candidateMapData.forEach((point) => {
      const band = getMapPointBand(point)
      if (band && band in counts) counts[band] += 1
    })
    return counts
  }, [candidateMapData, getMapPointBand])

  const priorityClassCounts = useMemo(() => mapData.reduce((counts, point) => {
    const priority = point.WCHI_Priority || point.Priority
    if (priority in counts) counts[priority] += 1
    return counts
  }, { High: 0, Medium: 0, Low: 0 }), [mapData])

  const approximatePriorityRegions = useMemo(
    () => buildApproximatePriorityRegions(mapData),
    [mapData],
  )

  const filteredMapData = useMemo(
    () => candidateMapData.filter((point) => {
      const band = getMapPointBand(point)
      return band != null && highlightedMapBands[band]
    }),
    [candidateMapData, getMapPointBand, highlightedMapBands],
  )

  const selectedMapPoint = mapData.find((point) => String(point.Micro_Watershed_ID) === String(selectedId))
  const selectedStructureOptions = selectedMapPoint?.Structure_Suitability || []
  const selectedStructureSuitability = selectedStructureOptions.find((item) => item.Structure === selectedSuitabilityStructure) || selectedStructureOptions[0]
  const rankingRows = useMemo(() => {
    const query = rankingSearch.trim().toLowerCase()
    return rankingData
      .filter((row) => rankingPriority === 'All' || row.WCHI_Priority === rankingPriority)
      .filter((row) => !query || String(row.Micro_Watershed_ID).toLowerCase().includes(query))
      .sort((first, second) => numberOr(second[rankingSort]) - numberOr(first[rankingSort]))
      .map((row, index) => ({ ...row, Rank: index + 1 }))
  }, [rankingData, rankingPriority, rankingSearch, rankingSort])

  const scenarioBaseline = useMemo(() => {
    if (!selectedWatershed) return null
    return Object.fromEntries(['Rainfall', 'Recharge', 'Terrain', 'Drainage', 'Vegetation', 'Groundwater', 'Runoff']
      .map((name) => [`${name}_Score`, numberOr(selectedWatershed[`${name}_Score`])]))
  }, [selectedWatershed])

  const activeScenarioComponents = scenarioComponents || scenarioBaseline

  const requestMapFocus = (id) => {
    setSelectedId(id)
    setMapSearchQuery(String(id))
    setMapFocusRequest((request) => request + 1)
    setMapPriorityFilter('All')
    setMapStructureFilter('All')
    setHighlightedMapBands({ High: true, Medium: true, Low: true })
    setMapSearchError('')
    setActivePage('Map')
  }

  const handleWatershedSearch = (event) => {
    event.preventDefault()
    const query = watershedQuery.trim().toLowerCase()
    const idMatch = watershedOptions.find((row) => String(row.Micro_Watershed_ID).toLowerCase() === query)
    const priorityMatch = ['High', 'Medium', 'Low'].find((priority) => `${priority.toLowerCase()} priority` === query || priority.toLowerCase() === query)
    const structureMatch = mapStructures.slice(1).find((structure) => structure.toLowerCase() === query)
    const siteMatch = idMatch || mapData.find((point) => (
      (!priorityMatch || (point.WCHI_Priority || point.Priority) === priorityMatch)
      && (!structureMatch || point.Recommended_Conservation_Structure === structureMatch)
    ))
    const canSearchCategory = priorityMatch || structureMatch
    if (!idMatch && !canSearchCategory) {
      setSearchError('Search by watershed ID, High/Medium/Low priority, or a listed conservation structure.')
      return
    }
    setSearchError('')
    setMapPriorityFilter(priorityMatch || 'All')
    setMapStructureFilter(structureMatch || 'All')
    if (siteMatch) requestMapFocus(siteMatch.Micro_Watershed_ID)
    else setActivePage('Map')
  }

  const selectMapWatershed = (id) => {
    setSelectedId(id)
    setMapSearchQuery(String(id))
    setWatershedQuery(String(id))
    setSearchError('')
  }

  const openPriorityOnMap = (priority) => {
    setMapPriorityFilter(priority)
    setMapStructureFilter('All')
    setActivePage('Map')
  }

  const getMapPointColor = (point) => {
    const band = getMapPointBand(point)
    return band ? MAP_AREA_COLORS[band] : '#94a3b8'
  }

  const handleScenario = async () => {
    setScenarioError('')
    setScenarioResult(null)
    setScenarioLoading(true)
    try {
      const response = await fetch(`${API}/scenario`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ Micro_Watershed_ID: selectedId, components: activeScenarioComponents }),
      })
      const responseBody = await response.text()
      let payload = null
      if (responseBody.trim()) {
        try {
          payload = JSON.parse(responseBody)
        } catch {
          throw new Error(`Scenario request failed (HTTP ${response.status}): the backend returned an invalid response.`)
        }
      }
      if (!response.ok) {
        throw new Error(payload?.error || `Scenario simulation failed (HTTP ${response.status}).`)
      }
      if (!payload) {
        throw new Error('The scenario service returned an empty response. Check that the Flask backend is running and try again.')
      }
      setScenarioResult(payload)
    } catch (error) {
      setScenarioError(error instanceof TypeError
        ? 'Could not reach the scenario service. Check that the Flask backend is running and try again.'
        : error.message)
    } finally {
      setScenarioLoading(false)
    }
  }

  const handleBatchUpload = async (event) => {
    const file = event.target.files?.[0]
    setUploadError('')
    setUploadResult(null)
    if (!file) return
    setUploadLoading(true)
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch(`${API}/batch-predict`, { method: 'POST', body })
      if (!response.ok) {
        const error = await response.json()
        const detail = error.missing_columns?.length ? ` Missing columns: ${error.missing_columns.join(', ')}` : ''
        throw new Error(`${error.error || 'Batch prediction failed.'}${detail}`)
      }
      const blob = await response.blob()
      const downloadUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = 'wchi_batch_predictions.csv'
      link.click()
      URL.revokeObjectURL(downloadUrl)
      setUploadResult('Predictions completed. The result CSV has been downloaded; the uploaded source file was not modified.')
    } catch (error) {
      setUploadError(error.message)
    } finally {
      setUploadLoading(false)
      event.target.value = ''
    }
  }

  const handlePredict = async () => {
    setPredictionError('')
    setPredictionResult(null)
    setPredictionLoading(true)
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formValues),
      })
      const contentType = response.headers.get('content-type') || ''
      if (!response.ok) {
        const err = contentType.includes('application/json') ? await response.json() : { error: 'Prediction failed' }
        throw new Error(err.error || 'Prediction failed')
      }
      const data = await response.json()
      setPredictionResult(data)
    } catch (error) {
      setPredictionError(error.message)
    } finally {
      setPredictionLoading(false)
    }
  }

  useEffect(() => {
    if (predictionResult) {
      predictionResultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [predictionResult])

  const updateForm = (key, value) => {
    predictionFormTouched.current = true
    setFormValues((prev) => ({ ...prev, [key]: value === '' ? '' : Number(value) }))
  }

  const renderDashboard = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4 2xl:grid-cols-7">
        <StatCard label="Total Micro-Watersheds" value={dashboard.total_micro_watersheds || microWatersheds.length} tone="blue" onClick={() => openPriorityOnMap('All')} />
        <StatCard label="High Priority" value={dashboard.high_priority || 0} tone="red" onClick={() => openPriorityOnMap('High')} />
        <StatCard label="Medium Priority" value={dashboard.medium_priority || 0} tone="amber" onClick={() => openPriorityOnMap('Medium')} />
        <StatCard label="Low Priority" value={dashboard.low_priority || 0} tone="green" onClick={() => openPriorityOnMap('Low')} />
        <StatCard label="Average WCHI" value={dashboard.average_wchi == null ? '—' : dashboard.average_wchi.toFixed(3)} tone="indigo" />
        <StatCard label="Average Rainfall" value={dashboard.average_rainfall_mm == null ? '—' : `${dashboard.average_rainfall_mm.toFixed(1)} mm`} tone="blue" />
        <StatCard label="Average Recharge Score" value={dashboard.average_recharge_score == null ? '—' : dashboard.average_recharge_score.toFixed(3)} tone="green" />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Predicted WCHI Distribution">
          {predictionDataReady ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={wchiChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" interval={0} angle={-45} textAnchor="end" height={75} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="WCHI" fill="#2563eb" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <PredictionUnavailable />}
        </Panel>
        <Panel title="Predicted Priority Distribution">
          {predictionDataReady ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={priorityDistribution} dataKey="value" nameKey="name" innerRadius={45} outerRadius={80} label>
                  {priorityDistribution.map((entry) => (
                    <Cell key={entry.name} fill={priorityColors[entry.name] || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : <PredictionUnavailable />}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Rainfall vs WCHI">
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Rainfall" />
              <YAxis type="number" dataKey="y" name="WCHI" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter data={predictedWatersheds.map((row) => ({ x: row.Predicted_Rainfall_Score, y: row.Predicted_WCHI_Score, name: row.Micro_Watershed_ID }))} fill="#2563eb" />
            </ScatterChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Recharge vs WCHI">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={predictedWatersheds}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Micro_Watershed_ID" interval={0} angle={-45} textAnchor="end" height={75} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="Predicted_Recharge_Score" stroke="#16a34a" />
              <Line type="monotone" dataKey="Predicted_WCHI_Score" stroke="#2563eb" />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Observed monthly rainfall">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={monthlyAnalytics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Month" minTickGap={28} />
              <YAxis /><Tooltip />
              <Line type="monotone" dataKey="Rainfall_mm" name="Mean rainfall (mm)" stroke="#167c80" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Observed monthly index and runoff">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={monthlyAnalytics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Month" minTickGap={28} />
              <YAxis /><Tooltip /><Legend />
              <Line type="monotone" dataKey="WCHI_Score" name="Mean WCHI" stroke="#155e75" dot={false} />
              <Line type="monotone" dataKey="Recharge_Score" name="Mean recharge score" stroke="#16825d" dot={false} />
              <Line type="monotone" dataKey="Runoff_mm" name="Mean runoff (mm)" stroke="#d79521" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="Recommended structure distribution across sites">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={structureSummary} layout="vertical" margin={{ left: 20, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="0" width={190} />
            <Tooltip formatter={(value) => [`${value} sites`, 'Recommended']} />
            <Bar dataKey="1" name="Sites" fill="#167c80" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  )

  const renderModelPerformance = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-4">
        {['Rainfall_Score', 'Recharge_Score', 'WCHI_Score', 'WCHI_Priority'].map((target) => {
          const row = metrics.filter((row) => row.Target === target)
          const validation = row.find((item) => item['Validation/Test'] === 'Validation') || row[0]
          return (
            <Panel key={target} title={target}>
              <div className="space-y-2 text-sm">
                <div><strong>Best Algorithm:</strong> {validation?.Algorithm || '—'}</div>
                {validation?.MAE !== undefined && <div>MAE: {Number(validation.MAE).toFixed(4)}</div>}
                {validation?.RMSE !== undefined && <div>RMSE: {Number(validation.RMSE).toFixed(4)}</div>}
                {validation?.R2 !== undefined && <div>R²: {Number(validation.R2).toFixed(4)}</div>}
                {validation?.Accuracy !== undefined && <div>Accuracy: {Number(validation.Accuracy).toFixed(4)}</div>}
                {validation?.Precision !== undefined && <div>Precision: {Number(validation.Precision).toFixed(4)}</div>}
                {validation?.Recall !== undefined && <div>Recall: {Number(validation.Recall).toFixed(4)}</div>}
                {validation?.F1 !== undefined && <div>F1: {Number(validation.F1).toFixed(4)}</div>}
              </div>
            </Panel>
          )
        })}
      </div>

      <Panel title="Recorded validation and held-out test metrics">
        <div className="overflow-auto">
          <table className="min-w-[900px] text-sm">
            <thead><tr className="border-b border-slate-200 text-left">
              {['Target', 'Algorithm', 'Split', 'MAE', 'RMSE', 'R²', 'Accuracy', 'Precision', 'Recall', 'F1'].map((header) => <th key={header} className="p-2">{header}</th>)}
            </tr></thead>
            <tbody>{metrics.map((row, index) => (
              <tr key={`${row.Target}-${row['Validation/Test']}-${index}`} className="border-b border-slate-100">
                <td className="p-2">{row.Target}</td><td className="p-2">{row.Algorithm}</td><td className="p-2">{row['Validation/Test']}</td>
                {['MAE', 'RMSE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1'].map((key) => <td key={key} className="p-2">{Number.isFinite(Number(row[key])) ? Number(row[key]).toFixed(4) : '—'}</td>)}
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-500">Training metrics were not stored in the existing outputs. No values are estimated or filled in here.</p>
      </Panel>

      <Panel title="Candidate algorithm comparison (validation only)">
        <div className="overflow-auto">
          <table className="min-w-[850px] text-sm">
            <thead><tr className="border-b border-slate-200 text-left">{['Target', 'Candidate algorithm', 'MAE', 'RMSE', 'R²', 'Accuracy', 'F1'].map((header) => <th key={header} className="p-2">{header}</th>)}</tr></thead>
            <tbody>{metrics.filter((row) => row['Validation/Test'] === 'Validation Candidate').map((row, index) => (
              <tr key={`${row.Target}-${row.Algorithm}-${index}`} className="border-b border-slate-100">
                <td className="p-2">{row.Target}</td><td className="p-2">{row.Algorithm}</td>
                {['MAE', 'RMSE', 'R2', 'Accuracy', 'F1'].map((key) => <td key={key} className="p-2">{Number.isFinite(Number(row[key])) ? Number(row[key]).toFixed(4) : '—'}</td>)}
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-500">Candidates are trained only on the supplied Train rows and compared on Validation. The Test split remains reserved for the saved selected models.</p>
      </Panel>

      <Panel title="Actual vs Predicted WCHI">
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="Predicted" domain={[0, 1]} />
            <YAxis type="number" dataKey="y" name="Actual" domain={[0, 1]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={scatterData} fill="#10b981" />
          </ScatterChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Prediction Table">
        <div className="mb-3 flex justify-end">
          <a href={`${API}/download/predictions`} download className="rounded bg-slate-800 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700">
            Download Predictions CSV
          </a>
        </div>
        <div className="max-h-[320px] overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left">
                <th className="p-2">Micro-Watershed</th>
                <th className="p-2">Actual</th>
                <th className="p-2">Predicted</th>
                <th className="p-2">Error</th>
                <th className="p-2">Priority</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 12).map((row, index) => (
                <tr key={`${row.Micro_Watershed_ID}-${index}`} className="border-b border-slate-100">
                  <td className="p-2">{row.Micro_Watershed_ID}</td>
                  <td className="p-2">{Number(row.Actual_WCHI_Score || 0).toFixed(4)}</td>
                  <td className="p-2">{Number(row.Predicted_WCHI_Score || 0).toFixed(4)}</td>
                  <td className="p-2">{Math.abs(Number(row.Actual_WCHI_Score || 0) - Number(row.Predicted_WCHI_Score || 0)).toFixed(4)}</td>
                  <td className="p-2">{row.Predicted_WCHI_Priority || row.Actual_WCHI_Priority}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )

  const renderPredictive = () => (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <label className="text-sm font-medium text-slate-700">
          Micro-watershed
          <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)} className="mt-1 block min-w-48 rounded border border-slate-300 bg-white px-3 py-2">
            {watershedOptions.map((row) => <option key={row.Micro_Watershed_ID} value={row.Micro_Watershed_ID}>{row.Micro_Watershed_ID}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => setActivePage('Conservation Structure Suitability')} className="rounded bg-sky-800 px-3 py-2 text-sm font-semibold text-white hover:bg-sky-900">
          Open suitability analysis
        </button>
      </div>
      {selectedWatershed && (
        <>
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatCard label="Observed mean WCHI" value={numberOr(selectedWatershed.WCHI_Score).toFixed(3)} tone="blue" />
            <StatCard label="Priority" value={selectedWatershed.WCHI_Priority || '—'} tone={selectedWatershed.WCHI_Priority === 'High' ? 'red' : selectedWatershed.WCHI_Priority === 'Medium' ? 'amber' : 'green'} />
            <StatCard label="Rainfall score" value={numberOr(selectedWatershed.Rainfall_Score).toFixed(3)} tone="blue" />
            <StatCard label="Recharge score" value={numberOr(selectedWatershed.Recharge_Score).toFixed(3)} tone="green" />
          </div>
          <Panel title="Environmental Profile">
            <div className="grid grid-cols-2 gap-3 text-sm md:grid-cols-3 xl:grid-cols-4">
              {[
                ['Rainfall', `${numberOr(selectedWatershed.Rainfall_mm).toFixed(1)} mm`],
                ['Recharge score', numberOr(selectedWatershed.Recharge_Score).toFixed(3)],
                ['Runoff', `${numberOr(selectedWatershed.Runoff_mm).toFixed(1)} mm`],
                ['Groundwater depth', `${numberOr(selectedWatershed.Groundwater_Depth_m).toFixed(1)} m`],
                ['Mean slope', `${numberOr(selectedWatershed.Mean_Slope_deg).toFixed(2)}°`],
                ['Drainage density', numberOr(selectedWatershed.Drainage_Density_km_per_km2).toFixed(2)],
                ['NDVI', numberOr(selectedWatershed.NDVI).toFixed(3)],
                ['NDWI', numberOr(selectedWatershed.NDWI).toFixed(3)],
                ['Soil pH', numberOr(selectedWatershed.Soil_pH).toFixed(2)],
                ['Infiltration', `${numberOr(selectedWatershed.Infiltration_Rate_mm_hr).toFixed(2)} mm/hr`],
                ['Forest cover', `${numberOr(selectedWatershed.Forest_pct).toFixed(1)}%`],
                ['Agriculture', `${numberOr(selectedWatershed.Agriculture_pct).toFixed(1)}%`],
              ].map(([label, value]) => <MetricBox key={label} label={label} value={value} />)}
            </div>
          </Panel>
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <Panel title="Why this WCHI priority?">
              <p className="mb-3 text-sm text-slate-600">{siteExplanation?.priority_explanation || 'Loading site explanation...'}</p>
              <div className="space-y-2">
                {(siteExplanation?.wchi_component_contributions || []).map((factor) => (
                  <div key={factor.name} className="grid grid-cols-[110px_1fr_56px] items-center gap-2 text-xs">
                    <span>{factor.name}</span>
                    <div className="h-2 rounded bg-slate-100"><div className="h-2 rounded bg-teal-700" style={{ width: `${factor.score * 100}%` }} /></div>
                    <span className="text-right">{factor.score.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs text-slate-500">Component contribution uses configured WCHI weights; this is an index decomposition, not a causal ML explanation.</p>
            </Panel>
            <Panel title="Conservation investigation">
              <div className="text-lg font-semibold text-slate-900">{selectedMapPoint?.Recommended_Conservation_Structure || '—'}</div>
              <p className="mt-2 text-sm text-slate-600">{siteExplanation?.recommended_structure?.Reason || 'Loading recommendation explanation...'}</p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <FactorList title="Supporting factors" factors={siteExplanation?.recommended_structure?.['Supporting Factors']} tone="green" />
                <FactorList title="Limiting factors" factors={siteExplanation?.recommended_structure?.['Limiting Factors']} tone="amber" />
              </div>
              <p className="mt-3 text-xs text-slate-500">Rule-based suitability, not an ML structure classifier. Field, hydrological, soil and engineering review is required before implementation.</p>
            </Panel>
          </div>
        </>
      )}
    </div>
  )

  const renderRankings = () => (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm font-medium text-slate-700">Search ID
          <input value={rankingSearch} onChange={(event) => setRankingSearch(event.target.value)} placeholder="e.g. SW12" className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2" />
        </label>
        <label className="text-sm font-medium text-slate-700">Priority
          <select value={rankingPriority} onChange={(event) => setRankingPriority(event.target.value)} className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2">
            {['All', 'High', 'Medium', 'Low'].map((value) => <option key={value}>{value}</option>)}
          </select>
        </label>
        <label className="text-sm font-medium text-slate-700">Sort by
          <select value={rankingSort} onChange={(event) => setRankingSort(event.target.value)} className="mt-1 block max-w-56 rounded border border-slate-300 bg-white px-3 py-2">
            {['WCHI_Score', 'Rainfall_Score', 'Recharge_Score', 'Runoff_Score', 'Groundwater_Score', 'Suitability Score'].map((value) => <option key={value} value={value}>{value.replace(/_/g, ' ')}</option>)}
          </select>
        </label>
        <span className="text-sm text-slate-500">{rankingRows.length} of {rankingData.length} sites</span>
      </div>
      <Panel title="Micro-Watershed Ranking">
        <div className="max-h-[70vh] overflow-auto">
          <table className="min-w-[1100px] text-sm">
            <thead className="sticky top-0 bg-white"><tr className="border-b border-slate-200 text-left">
              {['Rank', 'Micro-Watershed ID', 'WCHI Score', 'Priority', 'Rainfall', 'Recharge', 'Terrain', 'Drainage', 'Groundwater', 'Runoff', 'Recommended Structure', 'Suitability', 'Analysis'].map((header) => <th key={header} className="p-2">{header}</th>)}
            </tr></thead>
            <tbody>{rankingRows.map((row) => (
              <tr key={row.Micro_Watershed_ID} onClick={() => requestMapFocus(row.Micro_Watershed_ID)} className="cursor-pointer border-b border-slate-100 text-left hover:bg-sky-50">
                <td className="p-2 font-semibold">{row.Rank}</td><td className="p-2 font-medium">{row.Micro_Watershed_ID}</td>
                <td className="p-2">{numberOr(row.WCHI_Score).toFixed(3)}</td><td className="p-2">{row.WCHI_Priority}</td>
                <td className="p-2">{numberOr(row.Rainfall_Score).toFixed(2)}</td><td className="p-2">{numberOr(row.Recharge_Score).toFixed(2)}</td>
                <td className="p-2">{numberOr(row.Terrain_Score).toFixed(2)}</td><td className="p-2">{numberOr(row.Drainage_Score).toFixed(2)}</td>
                <td className="p-2">{numberOr(row.Groundwater_Score).toFixed(2)}</td><td className="p-2">{numberOr(row.Runoff_Score).toFixed(2)}</td>
                <td className="p-2">{row['Recommended Structure']}</td><td className="p-2">{numberOr(row['Suitability Score']).toFixed(1)}%</td>
                <td className="p-2">
                  <button type="button" onClick={(event) => {
                    event.stopPropagation()
                    setSelectedId(row.Micro_Watershed_ID)
                    setActivePage('Predictive Analysis')
                  }} className="rounded border border-slate-300 px-2 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100">
                    Analysis
                  </button>
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  )

  const renderScenario = () => (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm font-medium text-slate-700">Base watershed
          <select value={selectedId} onChange={(event) => { setSelectedId(event.target.value); setScenarioResult(null); setScenarioComponents(null) }} className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2">
            {watershedOptions.map((row) => <option key={row.Micro_Watershed_ID} value={row.Micro_Watershed_ID}>{row.Micro_Watershed_ID}</option>)}
          </select>
        </label>
        <span className="text-xs text-slate-500">Scenario only. Values are component scores from 0 to 1, not measured observations.</span>
      </div>
      {activeScenarioComponents && (
        <Panel title="Adjust WCHI components">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Object.entries(activeScenarioComponents).map(([key, value]) => (
              <label key={key} className="grid grid-cols-[1fr_64px] items-center gap-x-3 gap-y-2 text-sm font-medium text-slate-700">
                <span>{key.replace('_Score', '')}</span><span className="text-right tabular-nums">{numberOr(value).toFixed(2)}</span>
                <input type="range" min="0" max="1" step="0.01" value={value} onChange={(event) => setScenarioComponents((current) => ({ ...(current || scenarioBaseline), [key]: Number(event.target.value) }))} className="col-span-2 w-full accent-teal-700" />
              </label>
            ))}
          </div>
          <button type="button" onClick={handleScenario} disabled={scenarioLoading} className="mt-5 rounded bg-teal-800 px-4 py-2 text-sm font-semibold text-white hover:bg-teal-900 disabled:opacity-60">
            {scenarioLoading ? 'Simulating...' : 'Run scenario'}
          </button>
        </Panel>
      )}
      {scenarioError && <div role="alert" className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">{scenarioError}</div>}
      {scenarioResult && (
        <Panel title="Current vs simulated scenario">
          <p className="mb-4 text-sm text-amber-800">{scenarioResult.notice}</p>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {[['Current observation', scenarioResult.current], ['Modified scenario', scenarioResult.scenario]].map(([title, values]) => (
              <div key={title} className="rounded border border-slate-200 p-4">
                <h4 className="font-semibold">{title}</h4>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                  <MetricBox label="WCHI" value={numberOr(values.WCHI_Score).toFixed(3)} />
                  <MetricBox label="Priority" value={values.Priority} />
                  <MetricBox label="Top structure" value={values.Recommended_Structure} />
                  <MetricBox label="Suitability" value={`${values.Suitability}%`} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm font-medium">WCHI change: {scenarioResult.difference > 0 ? '+' : ''}{scenarioResult.difference} ({scenarioResult.percentage_change ?? '—'}%); suitability change: {scenarioResult.suitability_change > 0 ? '+' : ''}{scenarioResult.suitability_change} points.</p>
        </Panel>
      )}
    </div>
  )

  const renderExplainability = () => {
    const target = modelExplainability?.targets?.WCHI_Score
    const factors = target?.features || []
    return (
      <div className="space-y-5">
        <Panel title="Global WCHI model feature importance">
          <p className="mb-4 text-sm text-slate-600">{modelExplainability?.method || 'Loading model explanation...'}</p>
          <div className="space-y-3">
            {factors.slice(0, 15).map((factor) => (
              <div key={factor.Feature} className="grid grid-cols-[minmax(90px,180px)_1fr_60px] items-center gap-3 text-sm">
                <span className="truncate" title={factor.Feature}>{factor.Feature}</span>
                <div className="h-3 rounded bg-slate-100"><div className="h-3 rounded bg-teal-700" style={{ width: `${Math.min(100, factor.Importance)}%` }} /></div>
                <span className="text-right tabular-nums">{factor.Importance.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title={`Site explanation: ${selectedId || 'Select a watershed'}`}>
          <p className="text-sm text-slate-700">{siteExplanation?.priority_explanation || 'Select a watershed to load its WCHI component explanation.'}</p>
          <p className="mt-2 text-sm text-slate-600">Recommended investigation: {siteExplanation?.recommended_structure?.Structure || '—'} ({siteExplanation?.recommended_structure?.['Suitability Score'] ?? '—'}% rule-based suitability).</p>
          <FactorList title="Supporting factors" factors={siteExplanation?.recommended_structure?.['Supporting Factors']} tone="green" />
          <p className="mt-3 text-xs text-slate-500">Global feature importance describes model behavior across its training data; component contributions explain the configured WCHI calculation for this site. Neither is a causal explanation.</p>
        </Panel>
      </div>
    )
  }

  const renderFuturePrediction = () => (
    <div className="space-y-5">
      <Panel title="Future Prediction">
        <div className="rounded border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
          <strong>Future projection unavailable.</strong> {futureProjection?.reason || 'Checking the time series...' }
        </div>
        {futureProjection && <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricBox label="Observed months" value={futureProjection.observed_months ?? '—'} />
          <MetricBox label="First observation" value={futureProjection.date_start ?? '—'} />
          <MetricBox label="Last observation" value={futureProjection.date_end ?? '—'} />
        </div>}
      </Panel>
    </div>
  )

  const renderMethodology = () => (
    <div className="space-y-5">
      <Panel title="Analysis workflow">
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {(dashboard.methodology_steps || ['Dataset', 'Validation', 'Preprocessing', 'Feature selection', 'Model training', 'WCHI calculation', 'Priority classification', 'Structure suitability', 'GIS decision support']).map((step, index) => (
            <div key={step} className="flex items-center gap-3 rounded border border-slate-200 p-3 text-sm">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-800 text-xs font-bold text-white">{index + 1}</span>{step}
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Data, features and model boundaries">
        <div className="space-y-3 text-sm text-slate-700">
          <p>Dataset split is preserved as supplied: 336 Train, 72 Validation and 72 Test records. Site summaries average each watershed’s 24 monthly records; they are descriptive summaries, not extra training rows.</p>
          <p>Rainfall and recharge are predicted from the configured environmental model inputs. WCHI regression and priority classification use raw environmental input features and do not use WCHI or derived score labels as predictors.</p>
          <p>WCHI weights are configured in the backend. Priority thresholds are High at 0.70 and above, Medium from 0.40 to below 0.70, and Low below 0.40.</p>
          <p>Structure suitability is transparent rule-based analysis. Global feature importance is tree impurity importance; SHAP is not installed or claimed.</p>
        </div>
        <div className="mt-4 overflow-auto"><table className="w-full text-sm"><thead><tr className="border-b text-left"><th className="p-2">Target</th><th className="p-2">Model</th><th className="p-2">Input features</th></tr></thead><tbody>
          {Object.entries(modelExplainability?.targets || {}).map(([targetName, details]) => <tr key={targetName} className="border-b border-slate-100 align-top"><td className="p-2">{targetName}</td><td className="p-2">{details.model}</td><td className="p-2">{details.input_features.join(', ')}</td></tr>)}
        </tbody></table></div>
      </Panel>
      <AcademicDisclaimer />
    </div>
  )

  const renderMap = () => (
    <div className="space-y-6">
      <Panel title="Lower Vellar Micro-Watershed Visualization">
        <div className="mb-4 flex flex-wrap items-end gap-3">
          <label className="text-sm font-medium text-slate-700">Basemap
            <select
              value={mapBasemap}
              onChange={(event) => {
                setMapTileError(false)
                setMapBasemap(event.target.value)
              }}
              className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2"
            >
              <option value="dem">DEM terrain relief · OpenTopoMap</option>
              <option value="topo">Detailed topo · roads, water & land</option>
              <option value="street">Street map · OpenStreetMap</option>
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">Map layer
            <select value={mapLayer} onChange={(event) => setMapLayer(event.target.value)} className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2">
              <option value="Mean_Elevation_m">Mean elevation · site values</option>
              <option value="Mean_Slope_deg">Mean slope · site values</option>
              <option value="Drainage_Score">Drainage score · site values</option>
              <option value="Drainage_Density_km_per_km2">Drainage density · site values</option>
              <option value="WCHI_Score">WCHI score · site values</option>
              <option value="WCHI_Priority">WCHI priority</option>
              <option value="Rainfall_mm">Rainfall</option>
              <option value="Recharge_Score">Recharge score</option>
              <option value="Groundwater_Depth_m">Groundwater depth</option>
              <option value="Runoff_mm">Runoff</option>
              <option value="Structure_Suitability">Structure suitability</option>
              <option value="Conservation_Recommendation">Conservation recommendation</option>
              <option value="Existing_Ponds">Existing ponds</option>
              <option value="Existing_Check_Dams">Existing check dams</option>
            </select>
          </label>
          {mapLayer === 'Structure_Suitability' && (
            <label className="text-sm font-medium text-slate-700">Structure
              <select value={activeSuitabilityStructure} onChange={(event) => setSelectedSuitabilityStructure(event.target.value)} className="mt-1 block max-w-64 rounded border border-slate-300 bg-white px-3 py-2">
                {conservationStructures.map((structure) => <option key={structure}>{structure}</option>)}
              </select>
            </label>
          )}
          <label className="flex items-center gap-2 pb-2 text-sm font-medium text-slate-700">
            <input
              type="checkbox"
              checked={showMapMarkers}
              onChange={(event) => setShowMapMarkers(event.target.checked)}
              className="h-4 w-4 accent-teal-600"
            />
            Show watershed markers
          </label>
          {showMapMarkers && (mapLayer === 'Mean_Elevation_m' ? (
            <div className="min-w-64 flex-1 text-xs text-slate-600" aria-label="Watershed marker elevation legend">
              <div className="mb-1 flex items-center justify-between">
                <span>Low elevation</span>
                <span>Middle elevation</span>
                <span>High elevation</span>
              </div>
              <div
                className="h-3 rounded-full border border-slate-300/80 shadow-inner"
                role="img"
                aria-label="Map marker colors: green for low, yellow for middle, and red for high elevation"
                style={{ background: `linear-gradient(to right, ${MAP_AREA_COLORS.Low} 0 33.33%, ${MAP_AREA_COLORS.Medium} 33.33% 66.67%, ${MAP_AREA_COLORS.High} 66.67% 100%)` }}
              />
              {elevationRange && (
                <div className="mt-1 flex justify-between tabular-nums">
                  <span>{elevationRange.minimum.toFixed(0)} m</span>
                  <span>{(elevationRange.minimum + (elevationRange.maximum - elevationRange.minimum) / 3).toFixed(0)} m</span>
                  <span>{(elevationRange.minimum + 2 * (elevationRange.maximum - elevationRange.minimum) / 3).toFixed(0)} m</span>
                  <span>{elevationRange.maximum.toFixed(0)} m</span>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-wrap gap-3 text-xs text-slate-600" aria-label="Map area class legend">
              {MAP_AREA_BANDS.map(({ key, label, color }) => (
                <span key={key} className="flex items-center gap-1">
                  <i className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                  {label} {mapLayer === 'WCHI_Priority' ? 'priority' : 'values'}
                </span>
              ))}
            </div>
          ))}
          {mapBasemap === 'dem' && (
            <p className="text-xs text-slate-500">
              SRTM-based terrain relief with shaded elevation, contours, labels, and roads. Colored points show watershed priority: red for high, yellow for middle, and green for low.
            </p>
          )}
          <p className="map-data-note" role="note">
            Analysis colors represent watershed-level attributes from the supplied synthetic dataset at recorded site coordinates, not raster overlays or independently verified GIS processing. The shaded DEM-style basemap is for visual context only. No DEM raster, drainage-network geometry, or micro-watershed boundary files are available in this project, so those analytical surfaces and boundaries are not drawn. WCHI scores, priority classes, and rule-based conservation suitability are shown where calculated.
          </p>
        </div>
        <div className="mb-4 flex flex-wrap gap-3">
          {MAP_AREA_BANDS.map(({ key, label, color }) => (
            <div key={key} className="flex items-center gap-2 rounded border border-slate-200 px-3 py-2 text-sm">
              <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
              <span>{label}</span>
              <strong>{mapBandCounts[key]}</strong>
            </div>
          ))}
        </div>
        <div className="mb-4 rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-sm font-semibold text-slate-800">Highlight map areas</h4>
              <p className="text-xs text-slate-500">
                Choose which relative {mapLayer === 'WCHI_Priority' ? 'priority classes' : 'value classes'} to show for the selected map layer.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setHighlightedMapBands({ High: true, Medium: true, Low: true })}
              className="rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              Show all areas
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {MAP_AREA_BANDS.map(({ key, label, color }) => (
              <label
                key={key}
                className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm transition hover:border-slate-300"
              >
                <input
                  type="checkbox"
                  checked={highlightedMapBands[key]}
                  onChange={(event) => setHighlightedMapBands((current) => ({
                    ...current,
                    [key]: event.target.checked,
                  }))}
                  className="h-4 w-4"
                  style={{ accentColor: color }}
                  aria-label={`Show ${label.toLowerCase()} areas`}
                />
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="font-medium text-slate-700">{label} areas</span>
                <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs tabular-nums text-slate-600">
                  {mapBandCounts[key]}
                </span>
              </label>
            ))}
          </div>
          {mapLayer !== 'WCHI_Priority'
            && mapLayer !== 'Structure_Suitability'
            && mapLayer !== 'Conservation_Recommendation' && (
              <div className="mt-4 grid gap-3 border-t border-slate-200 pt-3 sm:grid-cols-2">
                <label className="map-band-slider-label">
                  <span className="flex justify-between gap-2 text-xs font-medium text-slate-700">
                    Low → Middle threshold
                    <strong>{mapBandBreaks.lower}%</strong>
                  </span>
                  <input
                    type="range"
                    min="1"
                    max={mapBandBreaks.upper - 1}
                    step="1"
                    value={mapBandBreaks.lower}
                    onChange={(event) => setMapBandBreaks((current) => ({
                      ...current,
                      lower: Number(event.target.value),
                    }))}
                    aria-label="Low to middle area threshold"
                    className="map-band-slider"
                  />
                </label>
                <label className="map-band-slider-label">
                  <span className="flex justify-between gap-2 text-xs font-medium text-slate-700">
                    Middle → High threshold
                    <strong>{mapBandBreaks.upper}%</strong>
                  </span>
                  <input
                    type="range"
                    min={mapBandBreaks.lower + 1}
                    max="99"
                    step="1"
                    value={mapBandBreaks.upper}
                    onChange={(event) => setMapBandBreaks((current) => ({
                      ...current,
                      upper: Number(event.target.value),
                    }))}
                    aria-label="Middle to high area threshold"
                    className="map-band-slider"
                  />
                </label>
                <p className="text-xs leading-5 text-slate-500 sm:col-span-2">
                  Thresholds are percentages of the selected layer’s observed range. WCHI priority and suitability layers retain their defined classes.
                </p>
              </div>
            )}
        </div>
        <div className="mb-4 flex flex-wrap items-end gap-3">
          <label className="text-sm font-medium text-slate-700">
            Priority
            <select value={mapPriorityFilter} onChange={(event) => setMapPriorityFilter(event.target.value)} className="mt-1 block rounded border border-slate-300 bg-white px-3 py-2">
              <option>All</option>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Structure
            <select value={mapStructureFilter} onChange={(event) => setMapStructureFilter(event.target.value)} className="mt-1 block max-w-[240px] rounded border border-slate-300 bg-white px-3 py-2">
              {mapStructures.map((structure) => <option key={structure}>{structure}</option>)}
            </select>
          </label>
          <button onClick={() => {
            setMapPriorityFilter('All')
            setMapStructureFilter('All')
            setHighlightedMapBands({ High: true, Medium: true, Low: true })
            setMapBandBreaks({ lower: 34, upper: 67 })
          }} className="rounded bg-slate-800 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700">
            Reset Map
          </button>
          <div className="ml-auto rounded bg-slate-100 px-3 py-2 text-sm text-slate-600">
            Showing {filteredMapData.length} of {mapData.length} locations
          </div>
        </div>
        <form
          className="map-search mb-3 flex flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            const query = mapSearchQuery.trim().toLowerCase()
            const match = mapData.find((point) => String(point.Micro_Watershed_ID).toLowerCase() === query)
            if (!match) {
              setMapSearchError('Enter an exact micro-watershed ID from the current dataset.')
              return
            }
            requestMapFocus(match.Micro_Watershed_ID)
            setMapSearchQuery(String(match.Micro_Watershed_ID))
          }}
        >
          <label className="min-w-56 flex-1 text-sm font-medium text-slate-700">
            Find a micro-watershed by ID
            <input
              value={mapSearchQuery}
              onChange={(event) => {
                setMapSearchQuery(event.target.value)
                setMapSearchError('')
              }}
              list="map-watershed-ids"
              placeholder="Enter a dataset ID"
              className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2"
              aria-label="Search micro-watershed by ID"
            />
            <datalist id="map-watershed-ids">
              {mapData.map((point) => <option key={point.Micro_Watershed_ID} value={point.Micro_Watershed_ID} />)}
            </datalist>
          </label>
          <button type="submit" className="map-search-button rounded-lg border px-4 py-2 text-sm font-semibold">
            Locate
          </button>
          {mapSearchError && <p className="w-full text-sm text-rose-600" role="alert">{mapSearchError}</p>}
        </form>
        <p className="mb-3 text-xs text-slate-500">
          Shaded zones estimate the nearest watershed's WCHI priority; they are illustrative and are not official watershed boundaries. Point colors show WCHI conservation priority: red for high, yellow for middle, and green for low.
        </p>
        {mapLayer === 'WCHI_Priority' && mapData.length > 0 && priorityClassCounts.Medium === mapData.length && (
          <p className="mb-3 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900" role="status">
            The dataset currently classifies all {priorityClassCounts.Medium} watershed locations as Medium. No High or Low priority areas are present under the configured WCHI thresholds.
          </p>
        )}
        <div ref={mapFrameRef} className="map-frame relative w-full overflow-hidden rounded-xl border border-slate-300 shadow-lg">
          <MapContainer
            center={mapView?.center || [11.5, 79.55]}
            zoom={mapView?.zoom ?? 9}
            minZoom={7}
            maxZoom={19}
            zoomControl={false}
            zoomSnap={0.5}
            zoomDelta={0.5}
            wheelPxPerZoomLevel={80}
            wheelDebounceTime={40}
            scrollWheelZoom
            doubleClickZoom
            zoomAnimation
            markerZoomAnimation
            fadeAnimation
            className="h-full w-full"
          >
            <MapViewport points={mapData} hasSavedView={Boolean(mapView)} />
            <MapViewPersistence onViewChange={setMapView} />
            <MapSizeInvalidator frameRef={mapFrameRef} />
            <MapMaxZoom maxZoom={mapBasemap === 'dem' ? 17 : 19} />
            <ZoomControl position="topright" />
            <MapHomeButton points={mapData} />
            <MapFullscreenButton frameRef={mapFrameRef} />
            <ScaleControl position="bottomleft" imperial={false} />
            {mapBasemap === 'topo' ? (
              <TileLayer
                key="esri-world-topo"
                attribution='Tiles &copy; <a href="https://www.esri.com/">Esri</a> — Sources: Esri, HERE, Garmin, Intermap, GEBCO, USGS, FAO, NPS, NRCAN, GeoBase, IGN, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), OpenStreetMap contributors, and the GIS User Community'
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                maxNativeZoom={19}
                maxZoom={19}
                keepBuffer={2}
                updateWhenZooming={false}
                eventHandlers={{
                  tileerror: () => setMapTileError(true),
                  tileload: () => setMapTileError(false),
                }}
              />
            ) : mapBasemap === 'dem' ? (
              <TileLayer
                key="dem-terrain"
                attribution='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, SRTM | Map style &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)'
                url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                maxNativeZoom={17}
                maxZoom={17}
                keepBuffer={2}
                updateWhenZooming={false}
                eventHandlers={{
                  tileerror: () => setMapTileError(true),
                  tileload: () => setMapTileError(false),
                }}
              />
            ) : (
              <TileLayer
                key="street"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                maxNativeZoom={19}
                maxZoom={19}
                keepBuffer={2}
                updateWhenZooming={false}
                eventHandlers={{
                  tileerror: () => setMapTileError(true),
                  tileload: () => setMapTileError(false),
                }}
              />
            )}
            {mapLayer === 'WCHI_Priority' && approximatePriorityRegions.map((region) => (
              <Polygon
                key={`priority-region-${region.id}`}
                positions={region.positions}
                pathOptions={{
                  color: MAP_AREA_COLORS[region.priority] || '#64748b',
                  fillColor: MAP_AREA_COLORS[region.priority] || '#64748b',
                  fillOpacity: 0.18,
                  opacity: 0.45,
                  weight: 1,
                }}
              />
            ))}
            {showMapMarkers && (
              <WatershedMarkers
                points={filteredMapData}
                selectedId={selectedId}
                onSelect={selectMapWatershed}
                getPointColor={getMapPointColor}
              />
            )}
            <MapFocusController
              points={mapData}
              selectedId={selectedId}
              focusRequest={mapFocusRequest}
              maxZoom={mapBasemap === 'dem' ? 17 : 19}
            />
          </MapContainer>
          {mapTileError && (
            <div className="map-tile-error" role="status">
              Some map tiles could not be loaded. Check your internet connection or switch basemaps.
            </div>
          )}
        </div>
      </Panel>
      {selectedMapPoint && (
        <Panel title={`Conservation potential: ${selectedMapPoint.Micro_Watershed_ID}`}>
          <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricBox label="Priority range" value={selectedMapPoint.WCHI_Priority || selectedMapPoint.Priority || 'Unknown'} />
            <MetricBox label="WCHI score" value={numberOr(selectedMapPoint.WCHI_Score).toFixed(4)} />
            <MetricBox label="Latitude" value={numberOr(selectedMapPoint.Latitude).toFixed(5)} />
            <MetricBox label="Longitude" value={numberOr(selectedMapPoint.Longitude).toFixed(5)} />
          </div>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h4 className="font-semibold text-slate-800">Structure suitability for this watershed</h4>
            <span className="text-sm text-slate-500">
              Recommended: {selectedMapPoint.Recommended_Conservation_Structure || 'Not specified'}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {(selectedMapPoint.Structure_Suitability || []).map((item) => (
              <div key={item.Structure} className="rounded border border-slate-200 p-3">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <strong className="text-sm">{item.Structure}</strong>
                  <span className="shrink-0 text-sm font-semibold">{item['Suitability Score']}%</span>
                </div>
                <div className="mb-2 h-2 rounded bg-slate-100">
                  <div className="h-2 rounded bg-emerald-600" style={{ width: `${item['Suitability Score']}%` }} />
                </div>
                <div className="text-xs font-medium text-slate-700">{item['Suitability Level']}</div>
                <p className="mt-1 text-xs leading-5 text-slate-500">{item.Reason}</p>
                <FactorList title="Supporting" factors={item['Supporting Factors']} tone="green" />
                <FactorList title="Limitations" factors={item['Limiting Factors']} tone="amber" />
              </div>
            ))}
          </div>
          {(!selectedMapPoint.Structure_Suitability || selectedMapPoint.Structure_Suitability.length === 0) && (
            <p className="text-sm text-slate-500">Structure suitability is not available for this watershed.</p>
          )}
        </Panel>
      )}
    </div>
  )

  const renderSuitability = () => (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <label className="font-medium">Select Micro-Watershed:</label>
        <select value={selectedId || ''} onChange={(e) => setSelectedId(e.target.value)} className="rounded border border-slate-300 bg-white px-3 py-2">
          {watershedOptions.map((row) => (
            <option key={row.Micro_Watershed_ID} value={row.Micro_Watershed_ID}>{row.Micro_Watershed_ID}</option>
          ))}
        </select>
      </div>

      {selectedWatershed && (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <Panel title="Watershed Details">
            <div className="space-y-2 text-sm">
              <div><strong>Micro-Watershed ID:</strong> {selectedWatershed.Micro_Watershed_ID}</div>
              <div><strong>Region:</strong> Lower Vellar Sub-Basin</div>
              <div><strong>Location:</strong> {numberOr(selectedWatershed.Latitude).toFixed(6)}° N, {numberOr(selectedWatershed.Longitude).toFixed(6)}° E</div>
              <div><strong>Latitude:</strong> {selectedWatershed.Latitude}</div>
              <div><strong>Longitude:</strong> {selectedWatershed.Longitude}</div>
              <div><strong>Area:</strong> {selectedWatershed.Area_km2 ? `${numberOr(selectedWatershed.Area_km2).toFixed(2)} km²` : '—'}</div>
              <div><strong>Mean Elevation:</strong> {selectedWatershed.Mean_Elevation_m ? `${numberOr(selectedWatershed.Mean_Elevation_m).toFixed(2)} m` : '—'}</div>
              <div><strong>WCHI:</strong> {selectedWatershed.WCHI_Score}</div>
              <div><strong>Priority:</strong> {selectedWatershed.WCHI_Priority}</div>
              <div><strong>Rainfall Score:</strong> {selectedWatershed.Rainfall_Score}</div>
              <div><strong>Recharge Score:</strong> {selectedWatershed.Recharge_Score}</div>
              <div><strong>Terrain Score:</strong> {selectedWatershed.Terrain_Score}</div>
              <div><strong>Drainage Score:</strong> {selectedWatershed.Drainage_Score}</div>
              <div><strong>Runoff Score:</strong> {selectedWatershed.Runoff_Score}</div>
              <div><strong>Groundwater Score:</strong> {selectedWatershed.Groundwater_Score}</div>
              <div><strong>Mean Slope:</strong> {selectedWatershed.Mean_Slope_deg}</div>
              <div><strong>Infiltration Rate:</strong> {selectedWatershed.Infiltration_Rate_mm_hr}</div>
              <div><strong>Existing Ponds:</strong> {selectedWatershed.Existing_Ponds}</div>
              <div><strong>Existing Check Dams:</strong> {selectedWatershed.Existing_Check_Dams}</div>
            </div>
          </Panel>

          <Panel title="Rule-Based Conservation Suitability">
            <p className="mb-3 text-xs text-slate-500">Transparent suitability rules; this is not a trained structure classifier or confirmation of construction feasibility.</p>
            <label className="mb-4 block text-sm font-medium text-slate-700">Structure to assess
              <select value={activeSuitabilityStructure} onChange={(event) => setSelectedSuitabilityStructure(event.target.value)} className="mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2">
                {conservationStructures.map((structure) => <option key={structure}>{structure}</option>)}
              </select>
            </label>
            {selectedStructureSuitability && <div className="rounded border border-slate-200 p-3">
              <div className="mb-2 flex items-center justify-between gap-2"><strong>{selectedStructureSuitability.Structure}</strong><span className="text-lg font-semibold">{selectedStructureSuitability['Suitability Score']}%</span></div>
              <div className="mb-2 h-2 rounded bg-slate-100"><div className="h-2 rounded bg-teal-700" style={{ width: `${selectedStructureSuitability['Suitability Score']}%` }} /></div>
              <span className="text-xs font-semibold">{selectedStructureSuitability['Suitability Level']}</span>
              <p className="mt-2 text-sm text-slate-600">{selectedStructureSuitability.Reason}</p>
              <div className="mt-3 grid gap-3"><FactorList title="Supporting environmental factors" factors={selectedStructureSuitability['Supporting Factors']} tone="green" /><FactorList title="Limiting factors" factors={selectedStructureSuitability['Limiting Factors']} tone="amber" /></div>
            </div>}
          </Panel>

          <Panel title="SW Location in Study Region">
            <div className="h-[320px] overflow-hidden rounded-lg">
              <MapContainer
                center={[numberOr(selectedWatershed.Latitude, 11.5), numberOr(selectedWatershed.Longitude, 79.55)]}
                zoom={13}
                scrollWheelZoom={false}
                className="h-full w-full"
              >
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <CircleMarker
                  center={[numberOr(selectedWatershed.Latitude, 11.5), numberOr(selectedWatershed.Longitude, 79.55)]}
                  radius={12}
                  pathOptions={{ color: '#0f766e', fillColor: '#14b8a6', fillOpacity: 0.85 }}
                >
                  <Popup>
                    <strong>{selectedWatershed.Micro_Watershed_ID}</strong><br />
                    Lower Vellar Sub-Basin<br />
                    {numberOr(selectedWatershed.Latitude).toFixed(6)}° N, {numberOr(selectedWatershed.Longitude).toFixed(6)}° E
                  </Popup>
                </CircleMarker>
              </MapContainer>
            </div>
          </Panel>
        </div>
      )}

    </div>
  )

  const renderRecommendations = () => (
    <div className="space-y-4">
      <Panel title="Water Conservation Recommendations">
        <div className="mb-3 flex justify-end">
          <a href={`${API}/download/recommendations`} download className="rounded bg-slate-800 px-3 py-2 text-sm font-medium text-white hover:bg-slate-700">
            Download Recommendations CSV
          </a>
        </div>
        <div className="max-h-[600px] overflow-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left">
                <th className="p-2">Micro-Watershed</th>
                <th className="p-2">Priority</th>
                <th className="p-2">WCHI</th>
                <th className="p-2">Recommended Structure</th>
                <th className="p-2">Suitability</th>
                <th className="p-2">Reason</th>
              </tr>
            </thead>
            <tbody>
              {recommendations.map((row, index) => (
                <tr key={`${row['Micro_Watershed_ID']}-${index}`} className="border-b border-slate-100 align-top">
                  <td className="p-2">{row['Micro_Watershed_ID']}</td>
                  <td className="p-2">{row.Priority || row['WCHI_Priority']}</td>
                  <td className="p-2">{Number(row.WCHI || 0).toFixed(4)}</td>
                  <td className="p-2">{row['Recommended Structure']}</td>
                  <td className="p-2">{row.Suitability}</td>
                  <td className="p-2 text-slate-600">{row.Reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )

  const renderInsights = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard label="Highest WCHI" value={dashboard.highest_wchi_micro_watershed || '—'} tone="red" />
        <StatCard label="Best Recharge Area" value={dashboard.best_recharge_area || '—'} tone="blue" />
        <StatCard label="Check Dams" value={dashboard.recommended_check_dams || 0} tone="amber" />
        <StatCard label="Farm Ponds" value={dashboard.recommended_farm_ponds || 0} tone="green" />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Panel title="Highest Priority Watersheds">
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left">
                  <th className="p-2">Rank</th>
                  <th className="p-2">Micro-Watershed</th>
                  <th className="p-2">WCHI</th>
                  <th className="p-2">Priority</th>
                  <th className="p-2">Structure</th>
                </tr>
              </thead>
              <tbody>
                {topWatersheds.map((row, index) => (
                  <tr key={`${row.Micro_Watershed_ID}-${index}`} className="border-b border-slate-100">
                    <td className="p-2 font-semibold">{index + 1}</td>
                    <td className="p-2">
                      <button type="button" onClick={() => requestMapFocus(row.Micro_Watershed_ID)} className="font-medium text-teal-300 underline-offset-2 hover:underline">
                        {row.Micro_Watershed_ID}
                      </button>
                    </td>
                    <td className="p-2">{numberOr(row.WCHI_Score).toFixed(4)}</td>
                    <td className="p-2">{row.WCHI_Priority || '—'}</td>
                    <td className="p-2">{recommendationLookup.get(String(row.Micro_Watershed_ID))?.['Recommended Structure'] || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Average Component Scores">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={averageScores} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" width={90} />
              <Tooltip formatter={(value) => Number(value).toFixed(4)} />
              <Bar dataKey="value" fill="#0f766e" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="Recommended Conservation Structures">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {structureSummary.map(([structure, count], index) => (
            <div key={`${structure}-${index}`} className="flex items-center justify-between rounded border border-slate-200 bg-slate-50 p-3">
              <span className="text-sm font-medium text-slate-700">{structure}</span>
              <span className="rounded bg-emerald-100 px-2 py-1 text-sm font-bold text-emerald-800">{count}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  )

  const renderReportsAndBatch = () => (
    <div className="space-y-6">
      <Panel title="Batch prediction from CSV">
        <p className="mb-4 text-sm text-slate-600">Each row must contain every environmental model input. Invalid rows are rejected with the row number; the uploaded source is never changed.</p>
        <label className="block text-sm font-medium text-slate-700">Choose prediction CSV
          <input type="file" accept=".csv,text/csv" onChange={handleBatchUpload} disabled={uploadLoading} className="mt-2 block w-full rounded border border-slate-300 bg-white p-2" />
        </label>
        <details className="mt-3 text-sm text-slate-600">
          <summary className="cursor-pointer font-medium">Required input columns</summary>
          <p className="mt-2 break-words">{Object.keys(initialForm).join(', ')}</p>
        </details>
        {uploadLoading && <p role="status" className="mt-3 text-sm text-sky-800">Validating inputs and generating batch predictions...</p>}
        {uploadError && <p role="alert" className="mt-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">{uploadError}</p>}
        {uploadResult && <p role="status" className="mt-3 rounded border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">{uploadResult}</p>}
      </Panel>

      <div className="flex flex-wrap gap-3 print:hidden">
        <button type="button" onClick={() => window.print()} disabled={!selectedWatershed} className="rounded bg-slate-800 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-900 disabled:opacity-50">Print / Save site report as PDF</button>
        <a href={`${API}/download/recommendations`} download className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download recommendations CSV</a>
        <a href={`${API}/download/metrics`} download className="rounded border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">Download model metrics CSV</a>
      </div>

      {selectedWatershed && (
        <article id="print-report" className="report-document rounded-lg border border-slate-200 bg-white p-6">
          <header className="mb-5 border-b border-slate-300 pb-4">
            <p className="text-xs font-semibold uppercase text-teal-800">Lower Vellar Sub-Basin | WCHI Decision Support</p>
            <h3 className="mt-2 text-2xl font-bold">Micro-Watershed Assessment: {selectedWatershed.Micro_Watershed_ID}</h3>
            <p className="mt-1 text-sm text-slate-500">Report generated {new Date().toLocaleDateString()}</p>
          </header>
          <section className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricBox label="Latitude" value={numberOr(selectedWatershed.Latitude).toFixed(5)} />
            <MetricBox label="Longitude" value={numberOr(selectedWatershed.Longitude).toFixed(5)} />
            <MetricBox label="WCHI" value={numberOr(selectedWatershed.WCHI_Score).toFixed(3)} />
            <MetricBox label="Priority" value={selectedWatershed.WCHI_Priority} />
          </section>
          <section className="mt-5">
            <h4 className="mb-2 font-semibold">Environmental profile</h4>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm md:grid-cols-3">
              {['Rainfall_mm', 'Rainfall_Score', 'Recharge_Score', 'Runoff_mm', 'Groundwater_Depth_m', 'Mean_Elevation_m', 'Mean_Slope_deg', 'Drainage_Density_km_per_km2', 'NDVI', 'NDWI', 'Soil_pH', 'Infiltration_Rate_mm_hr', 'Agriculture_pct', 'Forest_pct', 'Existing_Ponds', 'Existing_Check_Dams'].map((key) => (
                <div key={key} className="border-b border-slate-100 py-1"><strong>{key.replace(/_/g, ' ')}:</strong> {numberOr(selectedWatershed[key]).toFixed(3)}</div>
              ))}
            </div>
          </section>
          <section className="mt-5">
            <h4 className="font-semibold">WCHI explanation</h4>
            <p className="mt-1 text-sm">{siteExplanation?.priority_explanation || 'WCHI priority is classified using the configured 0.40 / 0.70 thresholds.'}</p>
            <p className="mt-2 text-sm"><strong>Recommended investigation:</strong> {siteExplanation?.recommended_structure?.Structure || '—'} ({siteExplanation?.recommended_structure?.['Suitability Score'] ?? '—'}% rule-based suitability)</p>
            <p className="mt-1 text-sm">{siteExplanation?.recommended_structure?.Reason || ''}</p>
          </section>
          <section className="mt-5"><AcademicDisclaimer /></section>
        </article>
      )}
    </div>
  )

  const renderNewPrediction = () => (
    <div className="space-y-6">
      <Panel title="Predict New Micro-Watershed">
        <div className="prediction-workspace-grid">
          <section className="prediction-workspace-panel">
            <div className="prediction-workspace-heading">
              <div>
                <h4>Essential watershed inputs</h4>
                <p>Enter the key site and environmental values. Other model-required features use {formDefaultsSource || 'dataset defaults'}.</p>
              </div>
              <span>{corePredictionInputs.length} inputs</span>
            </div>
            <div className="prediction-input-grid">
              {corePredictionInputs.map(([key, label]) => (
                <label key={key} className="prediction-input-field flex flex-col gap-1.5 rounded-xl border border-slate-200 p-3 text-sm font-medium">
                  <span>{label}</span>
                  <input
                    type="number"
                    step="any"
                    value={formValues[key] ?? ''}
                    onChange={(event) => updateForm(key, event.target.value)}
                    aria-label={label}
                    className="rounded border border-slate-300 px-3 py-2"
                  />
                </label>
              ))}
            </div>
            <button
              type="button"
              onClick={handlePredict}
              disabled={predictionLoading}
              className="mt-4 w-full rounded bg-emerald-700 px-5 py-3 font-semibold text-white hover:bg-emerald-800 disabled:cursor-wait disabled:opacity-60"
            >
              {predictionLoading ? 'Calculating prediction...' : 'Predict Conservation Priority'}
            </button>
            <p className="mt-3 text-xs text-slate-500">
              Results include WCHI priority and a rule-based conservation structure suitability assessment.
            </p>
          </section>

          <section ref={predictionResultRef} className="prediction-workspace-panel prediction-output-panel scroll-mt-4" aria-live="polite">
            <div className="prediction-workspace-heading">
              <div>
                <h4>Prediction results</h4>
                <p>Predicted watershed scores, priority, and conservation structure suitability.</p>
              </div>
            </div>
            {predictionLoading && (
              <div role="status" className="prediction-state-message">
                Calculating watershed health, conservation priority, and suitable structures...
              </div>
            )}
            {predictionError && (
              <div role="alert" className="prediction-state-message prediction-state-error">
                Prediction could not be completed: {predictionError}
              </div>
            )}
            {!predictionResult && !predictionLoading && !predictionError && (
              <div className="prediction-empty-state">
                <span aria-hidden="true">↗</span>
                <strong>Your prediction will appear here</strong>
                <p>Enter or adjust the watershed inputs, then run the prediction to view scores and structure comparisons.</p>
              </div>
            )}
            {predictionResult && (
              <>
                <div className="prediction-feature mb-4 grid overflow-hidden rounded-2xl border border-slate-200">
                  <StructureIllustration structure={predictionResult['Recommended Structure']} />
                  <div className="flex flex-col justify-center p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-700">Recommended conservation structure</p>
                    <h5 className="mt-2 text-xl font-bold text-slate-900">
                      {predictionResult['Recommended Structure'] || 'Site assessment'}
                    </h5>
                    <p className="mt-2 text-sm leading-6 text-slate-600">
                      {predictionResult['Recommendation Reason'] || 'A suitable water-conservation option based on the watershed inputs.'}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <span className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-800">
                        {predictionResult['Suitability Level'] || 'Suitability'} · {predictionResult['Structure Suitability'] ?? '—'}%
                      </span>
                      <span className="rounded-full border border-slate-200 bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                        {predictionResult.Priority || 'Priority'} priority
                      </span>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <MetricBox label="Rainfall score" value={predictionResult['Predicted Rainfall Score']} />
                  <MetricBox label="Recharge score" value={predictionResult['Predicted Recharge Score']} />
                  <MetricBox label="WCHI score" value={predictionResult['Predicted WCHI Score']} />
                  <MetricBox label="Priority" value={predictionResult.Priority} />
                </div>
                <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">Suitability factors</div>
                  <div className="mt-3 grid gap-3">
                    <FactorList title="Supporting factors" factors={predictionResult['Supporting Factors']} tone="green" />
                    <FactorList title="Limiting factors" factors={predictionResult['Limiting Factors']} tone="amber" />
                  </div>
                  <p className="mt-3 text-xs text-slate-500">{predictionResult['Recommendation Method']} Suitability is not a probability of construction success; confirm through field and engineering assessment.</p>
                  {predictionResult['Priority Model Confidence'] != null && (
                    <p className="mt-2 text-xs text-slate-500">Priority-model diagnostic confidence: {predictionResult['Priority Model Confidence']}%. {predictionResult['Priority Model Confidence Note']}</p>
                  )}
                  {predictionResult['Priority Model Prediction'] !== predictionResult.Priority && (
                    <p className="mt-2 text-xs text-amber-800">The classifier label differs from the threshold-based WCHI class; reported priority follows the configured WCHI thresholds.</p>
                  )}
                </div>
                <section className="prediction-options mt-4" aria-labelledby="prediction-options-title">
                  <div className="mb-3">
                    <h5 id="prediction-options-title" className="text-base font-semibold text-slate-100">Structure comparison</h5>
                    <p className="mt-1 text-xs text-slate-400">Scores are rule-based suitability comparisons, not probabilities of success.</p>
                  </div>
                  <div className="grid items-start gap-3">
                    {[...(predictionResult['Structure Suitability Options'] || [])]
                      .sort((first, second) => second['Suitability Score'] - first['Suitability Score'])
                      .map((item, index) => (
                        <article
                          key={item.Structure}
                          className={`prediction-option-card rounded-xl border p-4 text-sm ${item.Structure === predictionResult['Recommended Structure'] ? 'prediction-option-card-recommended' : ''}`}
                        >
                          <div className="flex items-start gap-3">
                            <StructureIllustration structure={item.Structure} compact />
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-start justify-between gap-2">
                                <div className="flex items-center gap-2">
                                  <span className="prediction-option-rank" aria-label={`Rank ${index + 1}`}>{index + 1}</span>
                                  <strong>{item.Structure}</strong>
                                </div>
                                <span className="prediction-option-score">{item['Suitability Score']}%</span>
                              </div>
                              <p className="mt-1 text-xs text-slate-400">{item['Suitability Level']}</p>
                            </div>
                          </div>
                          <div className="prediction-option-meter mt-3" role="progressbar" aria-label={`${item.Structure} suitability`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={item['Suitability Score']}>
                            <span style={{ width: `${Math.max(0, Math.min(100, item['Suitability Score']))}%` }} />
                          </div>
                          {item.Structure === predictionResult['Recommended Structure'] && (
                            <span className="prediction-option-recommended-label mt-3 inline-flex rounded-full px-2.5 py-1 text-xs font-semibold">Top recommendation</span>
                          )}
                          <p className="mt-3 leading-5 text-slate-300">{item.Reason}</p>
                          <div className="prediction-option-factors mt-4 grid gap-3">
                            <FactorList title="Why it may fit" factors={item['Supporting Factors']} tone="green" />
                            <FactorList title="Things to check" factors={item['Limiting Factors']} tone="amber" />
                          </div>
                        </article>
                      ))}
                  </div>
                  <p className="mt-3 text-xs leading-5 text-slate-400">
                    Shortlisted structures require field assessment and engineering review before implementation.
                  </p>
                </section>
              </>
            )}
          </section>
        </div>
      </Panel>
    </div>
  )

  const renderNavigationItem = (item) => (
    <button
      key={item}
      type="button"
      onClick={() => setActivePage(item)}
      aria-current={activePage === item ? 'page' : undefined}
      className={`w-auto shrink-0 whitespace-normal rounded-lg px-3 py-2.5 text-left text-sm transition md:w-full ${
        activePage === item ? 'bg-sky-600 text-white' : 'text-slate-200 hover:bg-slate-800'
      }`}
    >
      {item}
    </button>
  )

  return (
    <div className="app-shell min-h-screen bg-slate-100 text-slate-800">
      <div className="flex min-h-screen flex-col md:flex-row">
        <aside className="w-full min-w-0 bg-slate-900 text-white md:sticky md:top-0 md:h-screen md:w-[260px] md:shrink-0 md:overflow-y-auto">
          <div className="p-4 md:p-6">
            <h1 className="text-xl font-bold">WCHI Decision Support</h1>
            <p className="mt-2 text-sm text-slate-300">Lower Vellar Sub-Basin</p>
          </div>
          <nav aria-label="Main navigation" className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-x-visible md:pb-6">
            <div className="flex gap-1 md:flex-col">
              {primaryNavItems.map(renderNavigationItem)}
            </div>
            <details className="more-navigation" open={moreNavItems.includes(activePage) || undefined}>
              <summary>
                <span>More dashboards</span>
                <span className="more-navigation-count">{moreNavItems.length}</span>
              </summary>
              <div className="mt-1 flex flex-col gap-1">
                {moreNavItems.map(renderNavigationItem)}
              </div>
            </details>
          </nav>
        </aside>

        <main className="min-w-0 flex-1 p-4 sm:p-6">
          <header className="mb-6 flex flex-col gap-4 rounded-xl bg-white p-4 shadow-sm xl:flex-row xl:items-center xl:justify-between">
            <div className="min-w-0">
              <h2 className="text-2xl font-bold">GIS-Based Micro-Watershed Prioritization</h2>
              <p className="text-sm text-slate-500">Water Conservation Potential Index Mapping using WCHI</p>
            </div>
            <div className="flex min-w-0 flex-col gap-3 sm:flex-row xl:w-auto">
              <form onSubmit={handleWatershedSearch} className="flex min-w-0 flex-1 gap-2 xl:min-w-[340px]">
                <input
                  list="watershed-search-options"
                  value={watershedQuery}
                  onChange={(event) => { setWatershedQuery(event.target.value); setSearchError('') }}
                  placeholder="Find watershed ID"
                  aria-label="Find a micro-watershed"
                  className="min-w-0 flex-1 rounded border border-slate-300 px-3 py-2 text-sm outline-none focus:border-sky-600 focus:ring-2 focus:ring-sky-100"
                />
                <datalist id="watershed-search-options">
                  {watershedOptions.map((row) => <option key={row.Micro_Watershed_ID} value={row.Micro_Watershed_ID} />)}
                  {['High', 'Medium', 'Low'].map((priority) => <option key={priority} value={`${priority} priority`} />)}
                  {mapStructures.slice(1).map((structure) => <option key={structure} value={structure} />)}
                </datalist>
                <button type="submit" className="shrink-0 rounded bg-sky-700 px-3 py-2 text-sm font-semibold text-white hover:bg-sky-800">
                  Locate
                </button>
              </form>
              <div className="flex items-center justify-between gap-3">
                <div className="text-xs text-slate-500" aria-live="polite">
                  {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Connecting'}
                </div>
                <button
                  type="button"
                  onClick={refreshData}
                  disabled={loading}
                  className="rounded border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-wait disabled:opacity-60"
                >
                  {loading ? 'Refreshing...' : 'Refresh data'}
                </button>
              </div>
            </div>
            {searchError && <p role="alert" className="text-sm text-red-700 sm:basis-full">{searchError}</p>}
          </header>

          {loading && <div className="mb-6 rounded border border-sky-200 bg-sky-50 p-3 text-sm text-sky-800">Loading live watershed data...</div>}
          {loadError && (
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">
              <span>{loadError}</span>
              <button type="button" onClick={refreshData} className="rounded border border-red-300 px-3 py-1.5 font-semibold hover:bg-red-100">
                Retry connection
              </button>
            </div>
          )}

          {activePage === 'Dashboard' && renderDashboard()}
          {activePage === 'Predictive Analysis' && renderPredictive()}
          {activePage === 'Micro-Watershed Ranking' && renderRankings()}
          {activePage === 'Scenario Simulator' && renderScenario()}
          {activePage === 'AI Explainability' && renderExplainability()}
          {activePage === 'Map' && renderMap()}
          {activePage === 'Conservation Structure Suitability' && renderSuitability()}
          {activePage === 'Model Performance' && renderModelPerformance()}
          {activePage === 'Future Prediction' && renderFuturePrediction()}
          {activePage === 'Reports & Batch' && renderReportsAndBatch()}
          {activePage === 'Methodology' && renderMethodology()}
          {activePage === 'Water Conservation Recommendations' && renderRecommendations()}
          {activePage === 'Watershed Insights' && renderInsights()}
          {activePage === 'Predict New Micro-Watershed' && renderNewPrediction()}
        </main>
      </div>
    </div>
  )
}

function StructureIllustration({ structure = '', compact = false }) {
  const gradientId = useId().replace(/:/g, '')
  const name = structure.toLowerCase()
  const isWell = name.includes('well')
  const isPond = name.includes('pond') || name.includes('lake')
  const isDam = name.includes('dam')
  const isRecharge = name.includes('recharge') || name.includes('pit')
  const isBund = name.includes('bund')
  const isVegetation = name.includes('vegetative') || name.includes('soil conservation')
  const featureName = structure || 'Water conservation structure'

  return (
    <svg
      className={compact ? 'structure-illustration structure-illustration-compact' : 'structure-illustration'}
      viewBox="0 0 640 280"
      role="img"
      aria-label={`Illustration of a ${featureName}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <linearGradient id={`${gradientId}-sky`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#314c57" />
          <stop offset="1" stopColor="#55736f" />
        </linearGradient>
        <linearGradient id={`${gradientId}-water`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#73c5bc" />
          <stop offset="1" stopColor="#287e83" />
        </linearGradient>
        <linearGradient id={`${gradientId}-earth`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9a805a" />
          <stop offset="1" stopColor="#5e5443" />
        </linearGradient>
      </defs>
      <rect width="640" height="280" fill={`url(#${gradientId}-sky)`} />
      <circle cx="514" cy="59" r="28" fill="#e7c87e" opacity="0.88" />
      <path d="M0 157 116 79l91 70 94-84 126 96 82-64 131 74v69H0Z" fill="#425f54" />
      <path d="M0 188 116 125l99 56 112-65 104 61 94-60 115 57v65H0Z" fill="#344f47" />
      <path d="M0 207q124-26 228 0t206-2q104-27 206 0v75H0Z" fill="#48594b" />
      <path d="M0 247q117-21 233 0t207-1q108-21 200 0v34H0Z" fill={`url(#${gradientId}-earth)`} />

      {isPond && (
        <>
          <path d="M153 211q49-20 105-8t110-1q42-5 74 14-31 32-92 35l-144 1q-40-7-53-41Z" fill={`url(#${gradientId}-water)`} stroke="#b4d4a0" strokeWidth="5" />
          <path d="M142 215q31 26 74 27m224-22q-35 23-70 27" fill="none" stroke="#c2a878" strokeWidth="11" strokeLinecap="round" />
          <path d="M214 222h129m-101 10h94" stroke="#c1e5d0" strokeWidth="3" opacity="0.75" />
        </>
      )}

      {isWell && (
        <>
          <path d="M251 177h123l-13 68H264Z" fill="#82765f" stroke="#d6c5a0" strokeWidth="5" />
          <ellipse cx="312" cy="178" rx="62" ry="18" fill="#c4ad7d" stroke="#ead7ad" strokeWidth="5" />
          <ellipse cx="312" cy="179" rx="43" ry="10" fill="#214c4d" />
          <path d="M257 154v-25h13v25m84 0v-25h13v25m-99-25h98" stroke="#d1bd8d" strokeWidth="7" strokeLinecap="round" />
          <path d="M312 130v38m-8 0a8 8 0 1 0 16 0" fill="none" stroke="#d9cfaf" strokeWidth="4" />
        </>
      )}

      {isDam && (
        <>
          <path d="M0 218q123-31 239-6 63 14 126 6t275-6v68H0Z" fill={`url(#${gradientId}-water)`} />
          <path d="m259 177 35 5 14 74h-65Z" fill="#b09a73" stroke="#e0c99d" strokeWidth="4" />
          <path d="m248 193 51 1m-48 16 52 1m-49 16 51 1" stroke="#75674f" strokeWidth="4" />
          <path d="M307 244q30 14 65 5m-69 13q38 11 75 2" fill="none" stroke="#a2dad0" strokeWidth="4" opacity="0.8" />
        </>
      )}

      {isRecharge && (
        <>
          <ellipse cx="321" cy="231" rx="70" ry="25" fill="#877451" stroke="#c7ae7c" strokeWidth="5" />
          <ellipse cx="321" cy="228" rx="49" ry="14" fill="#163d40" />
          <path d="M306 132h30v91h-30Z" fill="#a99b7c" stroke="#ded0ab" strokeWidth="4" />
          <path d="M321 116v55m-17-17 17 18 17-18m-17 45 17 18 17-18" fill="none" stroke="#9cdbb2" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        </>
      )}

      {isBund && (
        <>
          <path d="M49 216q91-60 185-1t190-1q74-42 158 2" fill="none" stroke="#cfb46f" strokeWidth="10" strokeLinecap="round" />
          <path d="M26 240q113-59 221 0t216-2q73-37 151 1" fill="none" stroke="#9d8855" strokeWidth="9" strokeLinecap="round" />
          <path d="M49 211q91-60 185-1t190-1q74-42 158 2" fill="none" stroke="#e3d095" strokeWidth="2" />
        </>
      )}

      {isVegetation && (
        <>
          {[150, 230, 400, 490].map((x, index) => (
            <g key={x} transform={`translate(${x} ${index % 2 ? 10 : 24})`}>
              <path d="M35 220v-83" stroke="#735b43" strokeWidth="11" strokeLinecap="round" />
              <circle cx="35" cy="126" r="29" fill="#93a76d" />
              <circle cx="16" cy="141" r="22" fill="#607f59" />
              <circle cx="53" cy="144" r="22" fill="#789168" />
            </g>
          ))}
          <path d="M0 242q110-32 219 0t212-1q102-29 209 1" fill="none" stroke="#b3bd7a" strokeWidth="5" />
        </>
      )}

      {!isPond && !isWell && !isDam && !isRecharge && !isBund && !isVegetation && (
        <>
          <path d="M0 235q111-35 219 0t211-2q104-28 210 2" fill="none" stroke="#8bc7ad" strokeWidth="12" strokeLinecap="round" />
          <path d="M41 224q88-18 167 3m235-5q62-15 125 2" fill="none" stroke="#d4c08e" strokeWidth="5" strokeLinecap="round" />
          <circle cx="298" cy="211" r="23" fill="#b1a17b" />
          <path d="M272 215h53l-7 35h-40Z" fill="#837254" />
        </>
      )}

      <path d="M29 265q144-20 278 0t304-3" fill="none" stroke="#bda675" strokeWidth="2" opacity="0.55" />
      <path d="M27 255q144-20 278 0t304-3" fill="none" stroke="#bda675" strokeWidth="2" opacity="0.35" />
    </svg>
  )
}

function FactorList({ title, factors, tone = 'green' }) {
  const toneClasses = tone === 'amber' ? 'text-amber-900' : 'text-teal-900'
  const values = Array.isArray(factors) ? factors : []
  return (
    <div className="rounded border border-slate-200 p-3">
      <h5 className={`text-xs font-semibold ${toneClasses}`}>{title}</h5>
      <ul className="mt-2 list-inside list-disc space-y-1 text-xs leading-5 text-slate-600">
        {(values.length ? values : ['No factors reported.']).map((factor) => <li key={factor}>{factor}</li>)}
      </ul>
    </div>
  )
}

function AcademicDisclaimer() {
  return (
    <p className="rounded border border-amber-300 bg-amber-50 p-3 text-xs leading-5 text-amber-950">
      Academic decision-support only. These predictions and suitability results are not final engineering, hydrological, geological, environmental, or construction approval. Field verification, detailed site investigation, hydrological analysis, and approval by relevant authorities are required before implementation.
    </p>
  )
}

function Panel({ title, children }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="mb-4 text-lg font-semibold text-slate-800">{title}</h3>
      {children}
    </div>
  )
}

function StatCard({ label, value, tone, onClick }) {
  const colors = {
    blue: 'bg-blue-50 text-blue-700',
    red: 'bg-red-50 text-red-700',
    amber: 'bg-amber-50 text-amber-700',
    green: 'bg-green-50 text-green-700',
    indigo: 'bg-indigo-50 text-indigo-700',
  }
  const contents = (
    <>
      <div className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${colors[tone] || colors.blue}`}>{label}</div>
      <div className="mt-3 text-2xl font-bold text-slate-900">{value}</div>
    </>
  )
  const className = `w-full rounded-xl border border-slate-200 bg-white p-4 text-left shadow-sm ${onClick ? 'cursor-pointer transition hover:border-sky-300 hover:shadow-md focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-600' : ''}`
  return onClick
    ? <button type="button" onClick={onClick} className={className}>{contents}</button>
    : <div className={className}>{contents}</div>
}

function MetricBox({ label, value }) {
  return (
    <div className="rounded border border-slate-200 bg-slate-50 p-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-slate-800">{value ?? '—'}</div>
    </div>
  )
}

function PredictionUnavailable() {
  return <div className="flex h-[260px] items-center justify-center rounded bg-slate-50 text-sm text-slate-500">Prediction output is unavailable for this chart.</div>
}

function MapFullscreenButton({ frameRef }) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    const syncFullscreen = () => setIsFullscreen(document.fullscreenElement === frameRef.current)
    document.addEventListener('fullscreenchange', syncFullscreen)
    return () => document.removeEventListener('fullscreenchange', syncFullscreen)
  }, [frameRef])

  const toggleFullscreen = async () => {
    if (!frameRef.current) return
    if (document.fullscreenElement) await document.exitFullscreen()
    else await frameRef.current.requestFullscreen()
    setIsFullscreen(document.fullscreenElement === frameRef.current)
  }

  return (
    <button
      type="button"
      onClick={toggleFullscreen}
      aria-pressed={isFullscreen}
      className="map-fullscreen-button"
      aria-label={isFullscreen ? 'Exit fullscreen map' : 'Open fullscreen map'}
      title={isFullscreen ? 'Exit full screen' : 'Full screen'}
    >
      <span aria-hidden="true">{isFullscreen ? '⤢' : '⛶'}</span>
    </button>
  )
}

function MapHomeButton({ points }) {
  const map = useMap()

  const resetView = () => {
    const bounds = getMapBounds(points)
    if (bounds) map.flyToBounds(bounds, { padding: [32, 32], maxZoom: 10, duration: 0.8 })
    else map.flyTo([11.5, 79.55], 9, { duration: 0.8 })
  }

  return (
    <button
      type="button"
      onClick={resetView}
      className="map-home-button"
      aria-label="Reset to regional view"
      title="Reset to regional view"
    >
      <span aria-hidden="true">⌂</span>
    </button>
  )
}

function MapViewport({ points, hasSavedView }) {
  const map = useMap()
  const initialized = useRef(false)

  useEffect(() => {
    if (initialized.current || hasSavedView || points.length === 0) return
    initialized.current = true
    const bounds = getMapBounds(points)
    if (bounds) map.fitBounds(bounds, { padding: [32, 32], maxZoom: 10, animate: false })
  }, [map, points, hasSavedView])

  return null
}

function MapViewPersistence({ onViewChange }) {
  useMapEvents({
    moveend(event) {
      const map = event.target
      const center = map.getCenter()
      onViewChange({ center: [center.lat, center.lng], zoom: map.getZoom() })
    },
  })

  return null
}

function MapSizeInvalidator({ frameRef }) {
  const map = useMap()

  useEffect(() => {
    let frame = 0
    const invalidate = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => map.invalidateSize({ pan: false, debounceMoveend: true }))
    }
    const element = frameRef.current
    const observer = element ? new ResizeObserver(invalidate) : null
    if (element && observer) observer.observe(element)
    window.addEventListener('resize', invalidate)
    invalidate()

    return () => {
      cancelAnimationFrame(frame)
      observer?.disconnect()
      window.removeEventListener('resize', invalidate)
    }
  }, [map, frameRef])

  return null
}

function MapMaxZoom({ maxZoom }) {
  const map = useMap()

  useEffect(() => {
    map.setMaxZoom(maxZoom)
    if (map.getZoom() > maxZoom) map.setZoom(maxZoom, { animate: true })
  }, [map, maxZoom])

  return null
}

function MapFocusController({ points, selectedId, focusRequest, maxZoom }) {
  const map = useMap()

  useEffect(() => {
    if (!focusRequest || !selectedId) return
    const point = points.find((item) => String(item.Micro_Watershed_ID) === String(selectedId))
    if (!point) return
    const latitude = Number(point.Latitude)
    const longitude = Number(point.Longitude)
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return
    const targetZoom = Math.min(maxZoom, Math.max(14, map.getZoom()))
    if (document.visibilityState === 'visible') {
      map.flyTo([latitude, longitude], targetZoom, { duration: 0.9 })
    } else {
      map.setView([latitude, longitude], targetZoom, { animate: false })
    }
  }, [map, points, selectedId, focusRequest, maxZoom])

  return null
}

function getMapBounds(points) {
  const coordinates = points
    .map((point) => [Number(point.Latitude), Number(point.Longitude)])
    .filter(([latitude, longitude]) => (
      Number.isFinite(latitude) && Number.isFinite(longitude)
      && latitude >= -90 && latitude <= 90
      && longitude >= -180 && longitude <= 180
    ))
  return coordinates.length ? L.latLngBounds(coordinates) : null
}

function WatershedMarkers({ points, selectedId, onSelect, getPointColor }) {
  return points.map((point) => {
    const priority = point.WCHI_Priority || point.Priority || 'Low'
    const color = getPointColor(point)
    const selected = String(point.Micro_Watershed_ID) === String(selectedId)
    const options = Array.isArray(point.Structure_Suitability) ? point.Structure_Suitability : []
    const recommendation = point.Recommended_Conservation_Structure || '—'
    const suitability = options.find((item) => item.Structure === recommendation)?.['Suitability Score']

    return (
      <CircleMarker
        key={point.Micro_Watershed_ID}
        center={[Number(point.Latitude), Number(point.Longitude)]}
        radius={selected ? 9 : 7}
        eventHandlers={{
          click: (event) => {
            onSelect(point.Micro_Watershed_ID)
            event.target.openPopup()
          },
        }}
        pathOptions={{
          color: '#f8fafc',
          fillColor: color,
          fillOpacity: 0.9,
          weight: selected ? 2.5 : 1.5,
        }}
      >
        <MapTooltip direction="top" offset={[0, -7]} opacity={0.96}>
          <strong>{point.Micro_Watershed_ID}</strong> · {priority}
        </MapTooltip>
        <Popup>
          <div className="map-watershed-popup">
            <strong className="map-popup-title">{point.Micro_Watershed_ID}</strong>
            <dl>
              <div><dt>WCHI score</dt><dd>{point.WCHI_Score == null ? '—' : numberOr(point.WCHI_Score).toFixed(3)}</dd></div>
              <div><dt>Priority</dt><dd>{priority}</dd></div>
              <div><dt>Mean elevation</dt><dd>{point.Mean_Elevation_m == null ? '—' : `${numberOr(point.Mean_Elevation_m).toFixed(1)} m`}</dd></div>
              <div><dt>Mean slope</dt><dd>{point.Mean_Slope_deg == null ? '—' : `${numberOr(point.Mean_Slope_deg).toFixed(2)}°`}</dd></div>
              <div><dt>Drainage score</dt><dd>{point.Drainage_Score == null ? '—' : numberOr(point.Drainage_Score).toFixed(3)}</dd></div>
              <div><dt>Suitability</dt><dd>{suitability == null ? '—' : `${suitability}%`}</dd></div>
              <div><dt>Recommended structure</dt><dd>{recommendation}</dd></div>
              <div><dt>Location</dt><dd>{Number(point.Latitude).toFixed(5)}° N, {Number(point.Longitude).toFixed(5)}° E</dd></div>
            </dl>
          </div>
        </Popup>
      </CircleMarker>
    )
  })
}

export default App
