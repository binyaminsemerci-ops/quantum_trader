type ChartViewProps = {
  title?: string;
  children?: React.ReactNode;
};

export default function ChartView({ title, children }: ChartViewProps): JSX.Element {
  return (
    <section className="p-2 bg-slate-800 rounded">
      {title && <h2 className="text-sm font-semibold mb-2">{title}</h2>}
      <div>{children}</div>
    </section>
  );
}
