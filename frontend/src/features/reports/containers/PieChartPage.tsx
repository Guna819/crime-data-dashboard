import React, { useEffect, useMemo, useState } from "react";
import { Container, Row, Col, Form, Button, Card } from "react-bootstrap";
import { Bar, Doughnut } from "react-chartjs-2";
import Chart from "chart.js/auto";
import { BarChartDataResponse, CrimeFrequenceyType } from "types";
import useApiCall from "hooks/useApiCall";
import useApiCallLazy from "hooks/useApiCallLazy";

const PieChartPage = () => {

  const { fetchData } = useApiCallLazy<BarChartDataResponse>(
    `/crime/bar-chart`
  );
  const [data, setData] = useState<CrimeFrequenceyType[] | null>(null);

  useEffect(() => {
    fetchData().then((res) => {
      setData(res.data);
    });
  }, [fetchData]);

  const crimeLabels = useMemo(() => {
    return data?.map((item) => item.crime_type) || [];
  }, [data]);


  const pieChartData = useMemo(() => {
    const crimeData = crimeLabels.map((label) => {
      const item = data?.find((item) => item.crime_type === label);
      return item?.frequency || 0;
    })
    return{
      labels: crimeData,
      datasets: [
        {
          label: 'Crime Risk Levels',
          data: data?.map((item) => item.frequency) || [],
          backgroundColor: [
            'rgba(75, 192, 192, 0.2)',
            'rgba(255, 206, 86, 0.2)',
            'rgba(255, 99, 132, 0.2)',
            'rgba(153, 102, 255, 0.2)',
          ],
          borderColor: [
            'rgba(75, 192, 192, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(153, 102, 255, 1)',
          ],
          borderWidth: 1,
        },
      ],
    }
  }, [data, crimeLabels]);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Predicted Crime Risk Levels',
      },
    },
  };

  return (
    <Container className="mt-5">
        <Card className="p-3 mt-4">
          <h5>Pie Chart</h5>
          <Doughnut data={pieChartData} options={options} />
        </Card>
      </Container>
  );
};

export default PieChartPage;
