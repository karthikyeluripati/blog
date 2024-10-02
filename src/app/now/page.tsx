import Link from 'next/link';
import TimeDisplay from './TimeDisplay';

export default function Now() {
  return (
    <div className="max-w-2xl mx-auto p-4">
      {/* <h1 className="text-3xl font-bold mb-4">Now</h1> */}
      {/* <p className="mb-4">This is a <Link href="/now" className="text-blue-500 hover:underline">now</Link> page</p> */}
      
      <TimeDisplay />
      
      <ul className="list-disc list-inside mb-4 space-y-2">
        <li>working on AI stuff @ <Link href="https://www.linkedin.com/company/flozdesign/" className="text-blue-500 hover:underline">Floz</Link></li>
        <li>attending <Link href="https://www.meetup.com/new-york-machine-learning-meetup-group/?eventOrigin=event_home_page" className="text-blue-500 hover:underline">ML meetups</Link> at NYC</li>
        <li>experimenting entity tracking with Small Language Models</li>
        <li>reading The One Thing by Gary Keller</li>
      </ul>
      
      <p className="text-sm text-gray-500">Last updated: Sep 25, 2024</p>
    </div>
  );
}