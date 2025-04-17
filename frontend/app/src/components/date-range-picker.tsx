"use client"

import * as React from "react"

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { CalendarIcon } from "lucide-react"
import { DateRange } from "react-day-picker"
import { cn } from "@/lib/utils"
import { format } from "date-fns"

interface DateRangePickerProps {
  value?: DateRange;
  onChange?: (date: DateRange | undefined) => void;
  placeholder?: string;
  className?: string;
  size?: 'sm' | 'default';
}

export function DateRangePicker({
  className,
  value,
  onChange,
  placeholder = "Pick a date range",
  size = 'default'
}: DateRangePickerProps) {
  return (
    <div className={cn("grid gap-2", className)}>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            id="date"
            variant={"outline"}
            size={size}
            className={cn(
              "w-full justify-start text-left font-normal",
              !value && "text-muted-foreground",
              size === 'sm' && "text-sm h-8"
            )}
          >
            <CalendarIcon className={cn("mr-2", size === 'sm' ? "h-3 w-3" : "h-4 w-4")} />
            {value?.from ? (
              value.to ? (
                <>
                  {format(value.from, size === 'sm' ? "MMM d, y" : "LLL dd, y")} -{" "}
                  {format(value.to, size === 'sm' ? "MMM d, y" : "LLL dd, y")}
                </>
              ) : (
                format(value.from, size === 'sm' ? "MMM d, y" : "LLL dd, y")
              )
            ) : (
              <span>{placeholder}</span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            initialFocus
            mode="range"
            defaultMonth={value?.from}
            selected={value}
            onSelect={onChange}
            numberOfMonths={size === 'sm' ? 1 : 2}
          />
        </PopoverContent>
      </Popover>
    </div>
  )
} 