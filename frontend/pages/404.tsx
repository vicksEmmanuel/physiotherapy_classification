import { useRouter } from 'next/router';
import React, { useEffect } from 'react';

const FourOhFour = () => {
  const route = useRouter();
  useEffect(() => {
    route.push('/home');
  }, []);

  return <></>;
};

export default FourOhFour;
