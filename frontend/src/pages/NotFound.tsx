import { Link } from 'react-router-dom'
import { Card, StateMessage } from '@/components/ui'

export default function NotFound() {
  return (
    <Card className="mx-auto mt-10 max-w-lg">
      <StateMessage kind="empty" title="This page doesn't exist" detail="Check the address, or head back to the overview." />
      <Link to="/" className="mx-auto mb-4 block w-fit text-sm text-iris hover:underline">Go to the overview</Link>
    </Card>
  )
}
