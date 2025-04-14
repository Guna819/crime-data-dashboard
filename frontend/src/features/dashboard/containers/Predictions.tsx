// CrimePredictionChart.tsx
import React from 'react';
import { Container, Card } from 'react-bootstrap';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend, Title);

const Predictions: React.FC = () => {
  const data = {
    labels: ['Low', 'Medium', 'High', 'Very High'],
    datasets: [
      {
        label: 'Crime Risk Levels',
        data: [25, 35, 25, 15],
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
  };

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
    <>
      <Container className="mt-5">
        <h2 className="text-center">Crime Prediction</h2>
        <Card className="p-3 mt-4">
          <h5>Predicted Crime Risk Levels</h5>
          <Doughnut data={data} options={options} />
        </Card>
      </Container>
      <footer className="text-center mt-5 mb-3 text-muted">
        <p>© 2025 Crime Data Analytics Dashboard. All Rights Reserved.</p>
      </footer>
    </>
  );
};

export default Predictions;
