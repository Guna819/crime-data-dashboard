import { Badge, Container, ListGroup } from "react-bootstrap";
import { Crime } from "types";

type CrimeListProps = {
    crimes: Crime[];
}

const CrimeList: React.FC<CrimeListProps> = ({ crimes }) => {
    return (
      <Container className="my-4">
        <h4 className="mb-3">Recent Reported Crimes</h4>
        <ListGroup>
          {crimes.map((crime) => (
            <ListGroup.Item key={crime.ID}>
              <div className="d-flex justify-content-between">
                <div>
                  <strong>{crime.PrimaryType}</strong> - {crime.Description}
                  <br />
                  <small className="text-muted">
                    Case #{crime.CaseNumber} &middot; {crime.Date}
                  </small>
                  <br />
                  <small>{crime.Block} &mdash; {crime.LocationDescription}</small>
                </div>
                <div className="text-end">
                  {crime.Arrest && <Badge bg="success" className="mb-1">Arrest</Badge>}<br />
                  {crime.Domestic && <Badge bg="danger">Domestic</Badge>}
                </div>
              </div>
            </ListGroup.Item>
          ))}
        </ListGroup>
      </Container>
    );
  };
  
  export default CrimeList;