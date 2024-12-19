'use client';

import { cancelEvaluationTask, type EvaluationTask, type EvaluationTaskSummary as EvaluationTaskSummaryType, getEvaluationTask } from '@/api/evaluations';
import { DangerousActionButton } from '@/components/dangerous-action-button';
import { DateFormat } from '@/components/date-format';
import { mutateEvaluationTasks } from '@/components/evaluations/hooks';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { Skeleton } from '@/components/ui/skeleton';
import Link from 'next/link';
import * as React from 'react';
import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, Label, Pie, PieChart, XAxis } from 'recharts';
import useSWR from 'swr';

export function EvaluationTaskInfo ({ evaluationTaskId }: { evaluationTaskId: number }) {
  const { data } = useSWR(`api.evaluation.tasks.${evaluationTaskId}`, () => getEvaluationTask(evaluationTaskId));

  if (data) {
    return <EvaluationTaskInfoDisplay task={data} />;
  } else {
    return <EvaluationTaskInfoSkeleton />;
  }
}

export function EvaluationTaskInfoSkeleton () {
  return (
    <div className="space-y-4">
      <div className="py-[0.25em] text-xl">
        <Skeleton className="block w-[7em] h-[1em] rounded-sm" />
      </div>
      <div className="space-y-2 pb-[2px]">
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[7em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[8em] h-[1em] rounded-sm" />
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="col-span-1 lg:col-span-1">
          <StatusPieChartSkeleton />
        </div>
        <div className="col-span-1 lg:col-span-1">
          <RagasMetricsChartSkeleton />
        </div>
      </div>
    </div>
  );
}

export function EvaluationTaskInfoDisplay ({ task: { summary, ...task } }: { task: EvaluationTask }) {
  const canCancel = summary.not_start > 0;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">{task.name}</h2>
      <div className="text-muted-foreground text-xs space-y-2">
        <div>Dataset: <Link className="text-foreground underline" href={`/evaluation/datasets/${task.dataset_id}`}>{task.dataset_id}</Link></div>
        <div>Created at: <DateFormat date={task.created_at} /></div>
        <div>Updated at: <DateFormat date={task.updated_at} /></div>
        <div>User ID: {task.user_id}</div>
        {canCancel && <div>
          <DangerousActionButton
            size='sm'
            variant='destructive'
            action={async () => {
              await cancelEvaluationTask(task.id);
              void mutateEvaluationTasks();
            }}
          >
            Cancel Task
          </DangerousActionButton>
        </div>}
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="col-span-1 lg:col-span-1">
          <StatusPieChart summary={summary} />
        </div>
        <div className="col-span-1 lg:col-span-1">
          <RagasMetricsChart summary={summary} />
        </div>
      </div>
    </div>
  );
}

const pieChartConfig = {
  visitors: {
    label: 'Visitors',
  },
  not_start: {
    label: 'Not Start',
    color: 'hsl(var(--muted))',
  },
  progressing: {
    label: 'Processing',
    color: 'hsl(var(--info))',
  },
  succeed: {
    label: 'Succeed',
    color: 'hsl(var(--success))',
  },
  errored: {
    label: 'Errored',
    color: 'hsl(var(--destructive))',
  },
} satisfies ChartConfig;

function StatusPieChart ({ summary }: { summary: Pick<EvaluationTaskSummaryType, 'not_start' | 'errored' | 'progressing' | 'succeed'> }) {
  const totalTasks = useMemo(() => {
    return summary.not_start + summary.errored + summary.progressing + summary.succeed;
  }, [summary]);

  const chartData = useMemo(() => {
    return [
      { status: 'not_start', tasks: summary.not_start, fill: 'hsl(var(--accent))' },
      { status: 'progressing', tasks: summary.progressing, fill: 'hsl(var(--info))' },
      { status: 'succeed', tasks: summary.succeed, fill: 'hsl(var(--success))' },
      { status: 'errored', tasks: summary.errored, fill: 'hsl(var(--destructive))' },
    ];
  }, [summary]);

  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-lg font-normal">Evaluation Items</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={pieChartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Pie
              data={chartData}
              dataKey="tasks"
              nameKey="status"
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
                          {totalTasks.toLocaleString()}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          Items
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

const color_placeholder = '#71717a40';

function StatusPieChartSkeleton () {
  return (
    <Card className="flex flex-col">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-lg font-normal">Evaluation Items</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={pieChartConfig}
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
                          Items
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

const chartConfig = {
  semantic_similarity: {
    label: 'Semantic Similarity',
    color: 'hsl(var(--chart-1))',
  },
  factual_correctness: {
    label: 'Factual Correctness',
    color: 'hsl(var(--chart-2))',
  },
} satisfies ChartConfig;

export function RagasMetricsChart ({ summary }: { summary: Pick<EvaluationTaskSummaryType, keyof EvaluationTaskSummaryType & (`${string}_correctness` | `${string}_similarity`)> }) {
  const chartData = useMemo(() => {
    return [
      { metrics: 'min', semantic_similarity: summary.min_semantic_similarity, factual_correctness: summary.min_factual_correctness },
      { metrics: 'max', semantic_similarity: summary.max_semantic_similarity, factual_correctness: summary.max_factual_correctness },
      { metrics: 'avg', semantic_similarity: summary.avg_semantic_similarity, factual_correctness: summary.avg_factual_correctness },
      { metrics: 'std', semantic_similarity: summary.std_semantic_similarity, factual_correctness: summary.std_factual_correctness },
    ];
  }, [summary]);

  return (
    <Card className="flex flex-col justify-between h-full">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-lg font-normal">Ragas Metrics</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0 flex items-center justify-center">
        <div className="w-max h-max flex-1">
          <ChartContainer className="max-h-[192px] mx-auto" config={chartConfig}>
            <BarChart accessibilityLayer data={chartData}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="metrics"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                tickFormatter={(value) => value.slice(0, 3)}
              />
              <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent indicator="dashed" />}
              />
              <Bar dataKey="semantic_similarity" fill="hsl(var(--chart-1))" radius={4} />
              <Bar dataKey="factual_correctness" fill="hsl(var(--chart-2))" radius={4} />
            </BarChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}

export function RagasMetricsChartSkeleton () {
  return (
    <Card className="flex flex-col justify-between h-full">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-lg font-normal">Ragas Metrics</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0 flex items-center justify-center">
        <div className="w-max h-max flex-1">
          <ChartContainer className="max-h-[192px] mx-auto" config={chartConfig}>
            <BarChart accessibilityLayer>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="metrics"
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                tickFormatter={(value) => value.slice(0, 3)}
              />
              <ChartTooltip
                cursor={false}
                content={<ChartTooltipContent indicator="dashed" />}
              />
              <Bar dataKey="semantic_similarity" fill="hsl(var(--chart-1))" radius={4} />
              <Bar dataKey="factual_correctness" fill="hsl(var(--chart-2))" radius={4} />
            </BarChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  );
}
