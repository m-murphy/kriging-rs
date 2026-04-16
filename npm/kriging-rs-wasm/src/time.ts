/**
 * Date / time conversion helpers for space-time kriging inputs.
 *
 * Space-time kriging expects a numeric `times` array (any monotone unit — days,
 * seconds, ms). This module provides `timesFromDates` / `datesFromTimes` to
 * convert between native `Date` objects and that numeric axis without each
 * caller reinventing `Date.getTime() / 86400000`.
 *
 * @module
 */

/** Time granularity supported by {@link timesFromDates} and {@link datesFromTimes}. */
export type TimeUnit = "ms" | "s" | "minutes" | "hours" | "days";

const UNIT_TO_MS: Record<TimeUnit, number> = {
  ms: 1,
  s: 1000,
  minutes: 60_000,
  hours: 3_600_000,
  days: 86_400_000,
};

const DEFAULT_EPOCH = new Date(0);

/**
 * Convert a list of `Date` objects into a numeric time axis relative to
 * `epoch`, expressed in `unit`. The default unit is `"days"` and the default
 * epoch is the Unix epoch.
 *
 * @example
 * ```ts
 * const t = timesFromDates([new Date("2024-01-01"), new Date("2024-02-01")]);
 * // t[1] - t[0] === 31   (days)
 * ```
 */
export function timesFromDates(
  dates: readonly Date[],
  unit: TimeUnit = "days",
  epoch: Date = DEFAULT_EPOCH
): Float64Array {
  const factor = UNIT_TO_MS[unit];
  const epochMs = epoch.getTime();
  const out = new Float64Array(dates.length);
  for (let i = 0; i < dates.length; i++) {
    out[i] = (dates[i].getTime() - epochMs) / factor;
  }
  return out;
}

/**
 * Inverse of {@link timesFromDates}: turn a numeric time axis back into `Date`
 * objects relative to `epoch`, interpreting each value in `unit`.
 *
 * @example
 * ```ts
 * const dates = datesFromTimes([0, 31], "days");
 * // dates[1].getTime() - dates[0].getTime() === 31 * 86_400_000
 * ```
 */
export function datesFromTimes(
  times: ArrayLike<number>,
  unit: TimeUnit = "days",
  epoch: Date = DEFAULT_EPOCH
): Date[] {
  const factor = UNIT_TO_MS[unit];
  const epochMs = epoch.getTime();
  const out: Date[] = new Array(times.length);
  for (let i = 0; i < times.length; i++) {
    out[i] = new Date(epochMs + times[i] * factor);
  }
  return out;
}
