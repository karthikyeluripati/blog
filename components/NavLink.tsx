'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface NavLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;  // Make className optional
}

const NavLink: React.FC<NavLinkProps> = ({ href, children, className = '' }) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  return (
    <Link 
      href={href} 
      className={`mr-4 text-gray-400${isActive ? 'text-gray-800' : ''}`}
    >
      {children}
    </Link>
  );
};

export default NavLink;