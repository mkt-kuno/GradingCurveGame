// ================================================================
// Game Constants — shared across modules
// ================================================================

export interface ParticleLevel {
  name: string;
  sieve: string;
  upperSieveMM: number;
  radius: number;
  color: string;
  strokeColor: string;
  score: number;
}

export const LEVELS: ParticleLevel[] = [
  { name: '砂',     sieve: '0.75mm',  upperSieveMM: 2,     radius: 22, color: '#FDF5E6', strokeColor: '#C8B89C', score: 1 },
  { name: '細礫',   sieve: '2mm',     upperSieveMM: 4.75,  radius: 30, color: '#F8EED8', strokeColor: '#C4B49C', score: 2 },
  { name: '中礫',   sieve: '4.75mm',  upperSieveMM: 9.5,   radius: 42, color: '#F5E6CC', strokeColor: '#B8A48A', score: 4 },
  { name: '中礫',   sieve: '9.5mm',   upperSieveMM: 19,    radius: 60, color: '#EAD5B8', strokeColor: '#A89070', score: 7 },
  { name: '粗礫',   sieve: '19mm',    upperSieveMM: 26.5,  radius: 81, color: '#DCC4A0', strokeColor: '#9A7C5A', score: 11 },
  { name: '粗礫',   sieve: '26.5mm',  upperSieveMM: 37.5,  radius: 105, color: '#D0B48E', strokeColor: '#8C6C46', score: 16 },
  { name: '粗礫',   sieve: '37.5mm',  upperSieveMM: 53,    radius: 132, color: '#C2A47A', strokeColor: '#7E5E38', score: 22 },
  { name: '粗礫',   sieve: '53mm',    upperSieveMM: 75,    radius: 162, color: '#B49468', strokeColor: '#6E5030', score: 29 },
  { name: '石分',   sieve: '75mm',    upperSieveMM: 100,   radius: 195, color: '#A28458', strokeColor: '#5E4228', score: 37 },
  { name: '石分',   sieve: '100mm+',  upperSieveMM: 150,   radius: 231, color: '#907448', strokeColor: '#4E3420', score: 46 },
];

export const SIEVE_SIZES = [0.75, 2, 4.75, 9.5, 19, 26.5, 37.5, 53, 75];

export const GAME_W = 858;
export const GAME_H = 925;
export const WALL_T = 33;
export const CL = WALL_T;
export const CR = GAME_W - WALL_T;
export const CB = GAME_H - WALL_T;
export const CONTAINER_W = CR - CL;
export const CONTAINER_H = CB - WALL_T;
export const DROP_Y = 72;
export const DANGER_Y = 176;

export const GRAVITY = 700;

export const ESTAR_PP = 10000;
export const ESTAR_PW = 10000;

export const KN_PP = 100000;
export const KN_PW = 100000;

// Silica sand (Toyoura sand) — conservative (less bouncy) parameters
// Restitution lowered from 0.35 to 0.25 (safer side of 0.3-0.5 range)
// Rolling friction increased toward upper literature bounds
export const MU_PP = 0.65;
export const MU_PW = 0.80;
export const REST_PP = 0.25;
export const REST_PW = 0.25;

export const MU_ROLL_PP = 0.20;
export const MU_ROLL_PW = 0.35;

export const SUB_STEPS = 12;
export const MAX_DELTA_RATIO = 0.06;
export const MAX_VEL = 3000;
export const MAX_OMEGA = 80;
export const VEL_DAMP = 0.999;
export const ANG_DAMP = 0.998;

export function beta(e: number): number {
  if (e <= 0) return 1;
  if (e >= 1) return 0;
  const ln = Math.log(e);
  return -ln / Math.sqrt(Math.PI * Math.PI + ln * ln);
}

export const BETA_PP = beta(REST_PP);
export const BETA_PW = beta(REST_PW);
