import React, { useEffect, useMemo, useState } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { Bar } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { BarChartDataResponse } from "types";
import useApiCall from "hooks/useApiCall";

const BarChartPage = () => {
  const [month, setMonth] = useState<string | undefined>(undefined);
  const [year, setYear] = useState<string | undefined>(undefined);

  const params = useMemo(() => {
    const params = new URLSearchParams();

    if (month) params.append('month', month.toString());
    if (year) params.append('year', year.toString());
    return params;
  }, [month, year]);

  const { data, error, loading } = useApiCall<BarChartDataResponse>(
    `/crime/bar-chart?${params.toString() ? params.toString() : ''}`
  );

  const handleFilterChange = () => {};

  return (
    <Container className="mt-4">
      <Row>
        <Col>
          <h1>Crime Data Bar Chart</h1>
        </Col>
      </Row>

      <Row className="mb-4">
        <Col md={4}>
          <Form.Group controlId="monthSelect">
            <Form.Label>Select Month</Form.Label>
            <Form.Control
              as="select"
              value={month}
              onChange={(e) => setMonth(e.target.value)}
            >
              <option value="">All Months</option>
              <option value="1">January</option>
              <option value="2">February</option>
              <option value="3">March</option>
              <option value="4">April</option>
              <option value="5">May</option>
              <option value="6">June</option>
              <option value="7">July</option>
              <option value="8">August</option>
              <option value="9">September</option>
              <option value="10">October</option>
              <option value="11">November</option>
              <option value="12">December</option>
            </Form.Control>
          </Form.Group>
        </Col>
        <Col md={4}>
          <Form.Group controlId="yearSelect">
            <Form.Label>Select Year</Form.Label>
            <Form.Control
              as="select"
              value={year}
              onChange={(e) => setYear(e.target.value)}
            >
              <option value="">All Years</option>
              <option value="2011">2011</option>
              <option value="2012">2012</option>
              <option value="2013">2013</option>
              <option value="2014">2014</option>
              <option value="2015">2015</option>
              <option value="2016">2016</option>
              <option value="2017">2017</option>
              <option value="2018">2018</option>
              <option value="2019">2019</option>
              <option value="2020">2020</option>
              <option value="2021">2021</option>
              <option value="2022">2022</option>
              <option value="2023">2023</option>
            </Form.Control>
          </Form.Group>
        </Col>
        
      </Row>

      <Row>
        <Col>
          {data?.data ? (
            <Bar
              data={{
                labels: data.data.map((x) => x.crime_type),
                datasets: [
                  {
                    label: "Frequency",
                    data: data.data.map((x) => x.frequency),
                    backgroundColor: "rgba(54, 162, 235, 0.2)",
                    borderColor: "rgb(54, 162, 235)",
                    borderWidth: 1,
                  },
                ],
              }}
              options={{ responsive: true }}
            />
          ) : (
            <p>Loading...</p>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default BarChartPage;
