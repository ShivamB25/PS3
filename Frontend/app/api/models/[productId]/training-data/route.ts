import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { productId: string } }
) {
  try {
    const productId = params.productId
    const filePath = path.join(process.cwd(), '..', 'model_registry', productId, 'training_data.csv')

    if (!fs.existsSync(filePath)) {
      return new NextResponse('Training data not found', { status: 404 })
    }

    const data = fs.readFileSync(filePath, 'utf-8')
    return new NextResponse(data, {
      headers: {
        'Content-Type': 'text/csv',
      },
    })
  } catch (error) {
    console.error('Error reading training data:', error)
    return new NextResponse('Internal Server Error', { status: 500 })
  }
}