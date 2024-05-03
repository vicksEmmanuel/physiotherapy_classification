import { Environment } from "@/modules/environment";
import { useUserProfileState } from "@/state/useUserProfileState";

const HeaderBlock = () => {
    const {user} = useUserProfileState()
    return (
        <>
        <title>
            Smartsheet {Environment.isProduction ? '' : ' (dev)'}
        </title>
        <link rel="icon" href="/favicon.ico" />
    </>
    )
}
export default HeaderBlock;