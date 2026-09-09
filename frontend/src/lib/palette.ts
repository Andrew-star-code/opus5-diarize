/**
 * Цвета говорящих. Те же значения, что в backend/app/services/palette.py —
 * чтобы черновик, показанный вживую, не перекрасился при открытии записи.
 */
export const SPEAKER_COLORS = [
  "#2563eb",
  "#e11d48",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#be185d",
  "#4d7c0f",
  "#b45309",
  "#4f46e5",
];

export const UNKNOWN_COLOR = "#8b9499";

export function colorFor(index: number): string {
  return SPEAKER_COLORS[index % SPEAKER_COLORS.length];
}

export function defaultSpeakerName(index: number): string {
  return `Спикер ${index + 1}`;
}

/**
 * В живом режиме говорящие приходят метками движка (SPEAKER_00 и далее).
 * Раздаём цвета по очереди появления: первым заговорил — первый цвет.
 */
export function assignLiveColors(labels: (string | null)[]) {
  const order = new Map<string, number>();
  for (const label of labels) {
    if (label && !order.has(label)) order.set(label, order.size);
  }
  return {
    color: (label: string | null) =>
      label && order.has(label) ? colorFor(order.get(label)!) : UNKNOWN_COLOR,
    name: (label: string | null) =>
      label && order.has(label)
        ? defaultSpeakerName(order.get(label)!)
        : "Говорящий",
    count: order.size,
  };
}
