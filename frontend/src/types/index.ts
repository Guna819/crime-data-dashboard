import { User } from "./user";

export type AuthResponse = {
  message?: string;
  access_token?: string;
  error?: string;
};

export type LoginResponse = AuthResponse & {
  user?: User;
};

export type SignupResponse = AuthResponse & {
  user?: User;
};

export type District = {
  number: number;
  name: string;
  neighborhoods: string;
  active: boolean;
}

export type DashboardData = {
  total_incidents: number;
  most_common_crime: string;
  highest_crime_district: string;
  crime_solve_rate_percent: number;
  recent_incidents: Crime[];
  unique_crime_types: string[];
  unique_districts: District[];
};

export type HeatmapData = [number, number, number];

export type HeatmapResponse = {
  data: HeatmapData[];
};

export type Crime = {
    ID: number;
    CaseNumber: string;
    Date: string;
    Block: string;
    IUCR: number;
    PrimaryType: string;
    Description: string;
    LocationDescription: string;
    Arrest: boolean;
    Domestic: boolean;
    Beat: number;
    District: number;
    Ward: number;
    CommunityArea: number;
    FBICode: string;
    XCoordinate: number;
    YCoordinate: number;
    Year: number;
    UpdatedOn: string;
    Latitude: number;
    Longitude: number;
    Location: string;
    HistoricalWards2003_2015: number;
    ZipCodes: number;
    CommunityAreas: number;
    CensusTracts: number;
    Wards: number;
    Boundaries_ZIPCodes: number;
    PoliceDistricts: number;
    PoliceBeats: number;
};

export type CrimeResponse = {
  data: Crime[];
};

export type CrimeFrequenceyType =  {
    crime_type: string;
    frequency: number;
}

// 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall

enum Season {
    WINTER = 1,
    SPRING = 2,
    SUMMER = 3,
    FALL = 4
}

export type SeasonCrimeScatterData = CrimeFrequenceyType & {
    season: Season;
}

export type BarChartDataResponse = {
  data: CrimeFrequenceyType[];
}

export type SeasonCrimeScatterDataResponse = {
  data: SeasonCrimeScatterData[];
}