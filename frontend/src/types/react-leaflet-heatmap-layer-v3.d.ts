declare module "react-leaflet-heatmap-layer-v3" {
  import { FC } from "react";
  import { MapLayerProps } from "react-leaflet";
  import { LatLng, LatLngExpression } from "leaflet";

  interface HeatmapLayerProps extends MapLayerProps {
    points: Float[][]; // Array of points to render the heatmap
    options?: object; // Optional heatmap options like radius, blur, etc.
    longitudeExtractor: (point: Float[]) => number;
    latitudeExtractor: (point: Float[]) => number;
    intensityExtractor: (point: Float[]) => number;
    fitBoundsOnLoad?: boolean;
    fitBoundsOnUpdate?: boolean;
    max?: number;
    min?: number;
    gradient?: object;
    radius?: number;
    blur?: number;
    maxZoom?: number;
    minOpacity?: number;
  }

  export const HeatmapLayer: FC<HeatmapLayerProps>;
}
