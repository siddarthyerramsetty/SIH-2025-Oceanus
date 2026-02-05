

// NEW: A simple type for the initial list of floats returned by /api/floats
export interface FloatLocation {
  id: string;
  latitude: number;
  longitude: number;
  lastReported: string;
}

// This is the full data type for a single float's details from /api/float/{id}
export interface MeasurementPoint {
  date: string;
  value: number | null;
  latitude: number;
  longitude: number;
}
export interface ArgoFloat {
  id: string;
  lastReported: string;
  latitude: number;
  longitude: number;
  temperature: MeasurementPoint[];
  salinity: MeasurementPoint[];
  pressure: MeasurementPoint[];
}

// --- NEW DATA FETCHING FUNCTIONS ---

// Function #1: Fetches the lightweight list of all float locations
export async function getFloatLocations(): Promise<FloatLocation[]> {
  try {
    console.log('🔗 Attempting to fetch from: http://127.0.0.1:8000/api/floats');
    const response = await fetch("http://127.0.0.1:8000/api/floats");
    console.log('📡 Response received:', response.status, response.statusText);
    
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('📊 Data parsed successfully:', data);
    return data;
  } catch (error) {
    console.error("❌ There was a problem fetching the Argo float locations:", error);
    return []; 
  }
}

// Function #2: Fetches the detailed time-series data for ONE float
export async function getFloatDetails(floatId: string): Promise<ArgoFloat | null> {
  try {
    const response = await fetch(`http://127.0.0.1:8000/api/float/${floatId}`);
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`There was a problem fetching details for float ${floatId}:`, error);
    return null;
  }
}