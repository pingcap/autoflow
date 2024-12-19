"use client"

import { useTheme } from "next-themes"
import { Toaster as Sonner } from "sonner"

type ToasterProps = React.ComponentProps<typeof Sonner>

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme = "system" } = useTheme()

  return (
    <Sonner
      theme={theme as ToasterProps["theme"]}
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-muted-foreground group-[.toast.success]:text-success-foreground/70 group-[.toast.warning]:text-warning-foreground/70 group-[.toast.info]:text-info-foreground/70 group-[.toast.error]:text-destructive-foreground/70",
          success: 'success group-[.toaster]:bg-success group-[.toaster]:text-success-foreground group-[.toaster]:border-success-foreground/30',
          warning: 'warning group-[.toaster]:bg-warning group-[.toaster]:text-warning-foreground group-[.toaster]:border-warning-foreground/30',
          info: 'info group-[.toaster]:bg-info group-[.toaster]:text-info-foreground group-[.toaster]:border-info-foreground/30',
          error: 'error group-[.toaster]:bg-destructive group-[.toaster]:text-destructive-foreground group-[.toaster]:border-destructive-foreground/30',
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }
