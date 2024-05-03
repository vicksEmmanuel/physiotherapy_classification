import { ReactNode, useEffect, useState } from 'react';
import Skeleton from 'react-loading-skeleton';

const Wait = ({children}: {children: ReactNode}) => {
    const [isLoading, setIsLoading] = useState(true);
    useEffect(() => {
        setInterval(() => {
            setIsLoading(false);
        }, 1000)
    }, []);

    return <>{isLoading? 
        <div className='px-5'>
            <Skeleton count={2} /> 
            <div className='mt-3' />
            <Skeleton count={2} />
        </div>: children}</>;
};

export default Wait;