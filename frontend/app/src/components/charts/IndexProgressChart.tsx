'use client';

import type { IndexProgress } from '@/api/rag';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { Skeleton } from '@/components/ui/skeleton';
import * as React from 'react';
import { type ReactNode, useMemo } from 'react';
import { Label, Pie, PieChart } from 'recharts';

const color_error = '#ef4444';
const color_succeed = '#22c55e';
const color_in_progress = '#3b82f6';
const color_pending = '#71717a';
const color_blank = '#71717a80';
const color_placeholder = '#71717a40';

const chartConfig = {
  total: {
    label: 'Total',
  },
  completed: {
    label: 'Completed',
    color: color_succeed,
  },
  pending: {
    label: 'Pending',
    color: color_pending,
  },
  running: {
    label: 'Running',
    color: color_in_progress,
  },
  failed: {
    label: 'Failed',
    color: color_error,
  },
  not_started: {
    label: 'Not Started',
    color: color_blank,
  },
} satisfies ChartConfig;

export function IndexProgressChart ({ title, description, label, data }: { title: string, description?: string, label: ReactNode, data: IndexProgress }) {
  const total = React.useMemo(() => {
    return Object.values(data).reduce((a, b) => a + b, 0);
  }, [data]);

  const chartData = useMemo(() => {

    return [
      { count: data.completed, state: 'Completed', fill: color_succeed },
      { count: data.failed, state: 'Failed', fill: color_error },
      { count: data.pending, state: 'Pending', fill: color_pending },
      { count: data.running, state: 'Running', fill: color_in_progress },
      { count: data.not_started, state: 'Not started', fill: color_blank },
    ];
  }, []);

  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Pie
              data={chartData}
              dataKey="count"
              nameKey="state"
              innerRadius={60}
              strokeWidth={5}
            >
              <Label
                content={({ viewBox }) => {
                  if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-foreground text-3xl font-bold"
                        >
                          {total.toLocaleString()}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          {label}
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </Pie>
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

export interface IndexProgressChartPlaceholderProps {
  title: string,
  label: ReactNode,
  description?: string
}

export function IndexProgressChartPlaceholder ({ title, label, description }: IndexProgressChartPlaceholderProps) {
  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <Pie
              animationDuration={0}
              data={[{ count: 1, state: '', fill: color_placeholder }]}
              dataKey="count"
              nameKey="state"
              innerRadius={60}
              strokeWidth={5}
            >
              <Label
                content={({ viewBox }) => {
                  if (viewBox && 'cx' in viewBox && 'cy' in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-muted-foreground text-3xl font-bold"
                        >
                          --
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          {label}
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </Pie>
          </PieChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
