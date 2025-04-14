import { Button, Col, Container, Form, Row } from "react-bootstrap";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Map, MarsStroke } from "lucide-react";
import { icon } from "leaflet";
import { HeatmapLayer } from "react-leaflet-heatmap-layer-v3";
import { useCallback, useEffect, useState } from "react";
import { DashboardData, HeatmapData, HeatmapResponse } from "types";
import useApiCall from "hooks/useApiCall";
import useApiCallLazy from "hooks/useApiCallLazy";

const heatmapData = [
  [37.7749, -122.4194, 50],
  [41.745882542, -87.597167639, 0.1],
  [51.106, -0.08, 0.5],
  [51.507, -0.07, 0.3],
  [51.505, -0.09, 0.1],
  [51.508, -0.07, 0.2],
  [51.509, -0.06, 0.7],
  [51.507, -0.08, 0.4],
  [51.504, -0.07, 0.6],
  [51.503, -0.09, 0.3],
  [51.506, -0.06, 0.2],

  [40.7128, -74.006, 100],
  [34.0522, -118.2437, 80],
  [41.8781, -87.6298, 70],
  [29.7604, -95.3698, 60],
  [42.3601, -71.0589, 50],
  [32.7157, -117.1611, 40],
  [39.9526, -75.1652, 30],
  [33.4484, -112.074, 20],
  [47.6062, -122.3321, 10],
  [38.9072, -77.0369, 5],

  [19.076, 72.8777, 50], // Mumbai
  [19.041, 73.0777, 10], // Mumbai
  [19.066, 73.8077, 20], // Mumbai
  [19.076, 73.3077, 30], // Mumbai
  [28.7041, 77.1025, 40], // Delhi
  [12.9716, 77.5946, 30], // Bangalore
  [22.5726, 88.3639, 20], // Kolkata
  [13.0827, 80.2707, 10], // Chennai
  [26.9124, 75.7873, 5], // Jaipur
  [17.385, 78.4867, 5], // Hyderabad
  [22.7196, 75.8577, 2], // Indore
  [19.076, 72.8777, 3],
];

const CrimeMaps: React.FC = () => {
  const [data, setData] = useState<HeatmapData[]>([]);
  const {
    data: dashboardData,
    error: dashboardError,
    loading: dashboardLoading,
  } = useApiCall<DashboardData>("/dashboard");
  const [crimeType, setCrimeType] = useState<string | undefined>(undefined);
  const [selectedDistrict, setSelectedDistrict] = useState<string | undefined>(undefined);
  const [startDate, setStartDate] = useState<string | undefined>(undefined);
  const [endDate, setEndDate] = useState<string | undefined>(undefined);

  const { fetchData } = useApiCallLazy<HeatmapResponse>("/heatmap");
  

  const heatmapOptions = {
    radius: 20,
    blur: 20,
    maxZoom: 8,
    minOpacity: 0.5,
    maxOpacity: 1,
  };

  const handleFilter = useCallback(() => {
    let obj: Record<string, string> = {};
    if (crimeType) {
      obj.crime_type = crimeType
    };
    if (startDate) obj.start_date = startDate;
    if (endDate) obj.end_date = endDate;
    if (selectedDistrict) obj.district = selectedDistrict;
    fetchData(obj).then((res) => {
      setData(res.data);
    });
  }, [crimeType, startDate, endDate, selectedDistrict, fetchData]);

  useEffect(() => {
    fetchData().then((res) => {
      setData(res.data);
    });
  }, [fetchData]);

  return (
    <Container className="mt-4">
      <h2>Crime Maps Page</h2>
      <Row className="mb-4 g-3">
        <Col md={4}>
          <Form.Group controlId="crimeTypeSelect">
            <Form.Label>Select Crime Type</Form.Label>
            <Form.Control
              as="select"
              value={crimeType}
              onChange={(e) => setCrimeType(e.target.value)}
            >
              <option value="">All</option>
              {dashboardData?.unique_crime_types.map((crimeType) => (
                <option value={crimeType}>{crimeType}</option>
              ))}
            </Form.Control>
          </Form.Group>
        </Col>

        <Col md={4}>
          <Form.Group controlId="crimeTypeSelect">
            <Form.Label>Start Date</Form.Label>
            <Form.Control
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            ></Form.Control>
          </Form.Group>
        </Col>

        <Col md={4}>
          <Form.Group controlId="crimeTypeSelect">
            <Form.Label>End Date</Form.Label>
            <Form.Control
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            ></Form.Control>
          </Form.Group>
        </Col>

        <Col md={4}>
          <Form.Group controlId="crimeTypeSelect">
            <Form.Label>District</Form.Label>
            <Form.Control
              as="select"
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
            >
              <option value="">All</option>
              {dashboardData?.unique_districts.map((district) => (
                <option value={district.number}>{district.name}</option>
              ))}
            </Form.Control>
          </Form.Group>
        </Col>

      </Row>
      <Row>
        <Col md={4}>
          <Button variant="primary" onClick={handleFilter}>
            Filter
          </Button>
        </Col>
      </Row>
      <Row style={{ marginTop: "24px" }}>
        <Col md={12}>
          <div
            className="map-container"
            style={{ height: "80vh", width: "100%" }}
          >
            <MapContainer
              center={[41.745882542, -87.597167639]}
              zoom={13}
              scrollWheelZoom={false}
              style={{ height: "100%", width: "100%" }}
            >
              <HeatmapLayer
                fitBoundsOnLoad
                fitBoundsOnUpdate
                points={data ?? []}
                longitudeExtractor={(point) => point[1]}
                latitudeExtractor={(point) => point[0]}
                key={Math.random() + Math.random()}
                intensityExtractor={(point) => parseFloat(point[2])}
                {...heatmapOptions}
              />

              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
            </MapContainer>
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default CrimeMaps;
