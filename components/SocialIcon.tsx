import React from 'react';
import Link from 'next/link';

interface SocialIconProps {
  href: string;
  ariaLabel: string;
  children: React.ReactNode;
}

const SocialIcon: React.FC<SocialIconProps> = ({ href, ariaLabel, children }) => (
  <Link 
    href={href} 
    target="_blank" 
    rel="noopener noreferrer" 
    aria-label={ariaLabel}
    className="text-slate-300 hover:text-white mr-4"
  >
    {children}
  </Link>
);

export default SocialIcon;