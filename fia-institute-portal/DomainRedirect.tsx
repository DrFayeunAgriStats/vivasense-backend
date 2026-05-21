import { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export const DomainRedirect = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Check if we're on vivasensestat.com domain
    const hostname = window.location.hostname;
    const isVivaSenseDomain = 
      hostname === 'vivasensestat.com' || 
      hostname === 'www.vivasensestat.com';

    // If on VivaSense domain and NOT already on /vivasense route
    if (isVivaSenseDomain && !location.pathname.startsWith('/vivasense')) {
      // Redirect to /vivasense
      navigate('/vivasense', { replace: true });
    }
  }, [navigate, location.pathname]);

  // This component doesn't render anything
  return null;
};
