import type React from "react"

export const Area: React.FC<any> = ({ children, ...props }) => {
  return <g {...props}>{children}</g>
}

export const CartesianGrid: React.FC<any> = (props) => {
  return <g {...props} />
}

export const Line: React.FC<any> = ({ children, ...props }) => {
  return <g {...props}>{children}</g>
}

export const ResponsiveContainer: React.FC<any> = ({ children, ...props }) => {
  return <div {...props}>{children}</div>
}

export const Tooltip: React.FC<any> = (props) => {
  return <g {...props} />
}

export const XAxis: React.FC<any> = (props) => {
  return <g {...props} />
}

export const YAxis: React.FC<any> = (props) => {
  return <g {...props} />
}

