import yaml
import os, sys, time
import dynamixel_sdk
import numpy as np

class DxlGripperInterface():
    def __init__(self, cfg):
        self.protocol = cfg
        
        self._gripper_id = self.protocol["device"]["GRIPPER_ID"]
        
        self._homed_gripper = False
        # When homed, 0 position is the fully open position
        # Maximum range (greater than zero) is the closed position
        # Current: negative is opening, positive is closing
        self._gripper_range = 0
    
    @property
    def gripper_id(self):
        return self._gripper_id
    
    @staticmethod
    def deg_to_position(deg):
        return int(deg / 360.0 * 4095)
    
    @staticmethod
    def position_to_deg(pos):
        return int(pos / 4095.0 * 360)
    
    @staticmethod
    def twos_compliment(val):
        return -(val & 0x80000000) | (val & 0x7fffffff)
    
    @staticmethod
    def twos_compliment_2byte(val):
        return -(val & 0x8000) | (val & 0x7fff)


    def init(self):
        self.portHandler = dynamixel_sdk.PortHandler(
            self.protocol["device"]["DEVICENAME"])
        self.packetHandler = dynamixel_sdk.PacketHandler(2.0) # set protocol version

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            # print("Failed to open the port")
            raise SystemError("Port failure")
        
        # Set port baudrate
        if self.portHandler.setBaudRate(
            self.protocol["device"]["BAUDRATE"]):
            print("Succeeded to set the baudrate")
        else:
            # print("Failed to set the baudrate")
            raise SystemError("Baudrate setting failure")
    
    def _write_byte(self, n_byte, ID, addr, val):
        if n_byte == 1:
            func = self.packetHandler.write1ByteTxRx
        elif n_byte == 2:
            func = self.packetHandler.write2ByteTxRx
        elif n_byte == 4:
            func = self.packetHandler.write4ByteTxRx
        else:
            raise ValueError("n_byte not recognized")
        dxl_comm_result, dxl_error = func(
            self.portHandler, ID, addr, val)
        
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            raise SystemError
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            raise SystemError
    
    def _read_byte(self, n_byte, ID, addr):
        if n_byte == 1:
            func = self.packetHandler.read1ByteTxRx
        elif n_byte == 2:
            func = self.packetHandler.read2ByteTxRx
        elif n_byte == 4:
            func = self.packetHandler.read4ByteTxRx
        else:
            raise ValueError("n_byte not recognized")
        value, dxl_comm_result, dxl_error = func(
            self.portHandler, ID, addr)

        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            raise SystemError
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            raise SystemError

        return value

    def _set_torque(self, ID, enable):
        self._write_byte(1, ID, 
            self.protocol["control_table"]["ADDR_TORQUE_ENABLE"], enable)
        if enable:
            print("DXL ID:{} has been successfully connected".format(ID))
        else:
            print("DXL ID:{} has been successfully disconnected".format(ID))

    def enable_torque(self):
        self._set_torque(self.gripper_id, 1)
    
    def disable_torque(self):
        self._set_torque(self.gripper_id, 0)
    
    def set_control_mode(self, ID, mode):
        if mode == "current":
            goal_mode = 0
        elif mode == "velocity":
            goal_mode = 1
        elif mode == "position":
            goal_mode = 3
        elif mode == "cbp":
            goal_mode = 5
        else:
            raise ValueError(f"Mode is not defined: {mode}")

        self._write_byte(1, ID, 
                    self.protocol["control_table"]["ADDR_CONTROL_MODE"], goal_mode)
        if mode == "position":
            self.write_position_control_PID(ID)
    
    def read_position(self, ID):
        return self.twos_compliment(self._read_byte(4, ID,
                self.protocol["control_table"]["ADDR_PRESENT_POSITION"]))

    def read_current(self, ID):
        return self.twos_compliment_2byte(self._read_byte(2, ID,
                self.protocol["control_table"]["ADDR_PRESENT_CURRENT"]))
    
    def read_control_mode(self, ID):
        control_mode = self._read_byte(1, ID,
                self.protocol["control_table"]["ADDR_CONTROL_MODE"])
        # print("current control mode: ", control_mode)
        return control_mode
    
    def read_homing_offset(self, ID):
        return self.twos_compliment(self._read_byte(4, ID, 
                self.protocol["control_table"]["ADDR_HOMING_OFFSET"]))
    
    def close(self):
        self.disable_torque(True, False)
        self.portHandler.closePort()

    ## Position control
    def write_goal_pos(self, ID, pos, degrees=False, wait=False):
        # set position control
        # self.set_control_mode(ID, position=True, enable_torque=True)
        # if ID == self.thumb_id:
        #     deg = np.clip(pos, *self.protocol["workspace"]["thumb"])
        # if ID == self.duo_id:
        #     deg = np.clip(pos, *self.protocol["workspace"]["duo"])
        if degrees:
            pos = self.deg_to_position(pos)
        self._write_byte(4, ID, 
            self.protocol["control_table"]["ADDR_GOAL_POSITION"], pos)
        # print("Set goal pos for ID:{} as {}".format(ID, pos))

        if wait:
            while not self.close_to_goal(ID, pos):
                time.sleep(0.2)
            # print("Goal reached")


    def write_position_control_PID(self, ID, P=250, I=0, D=30):
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_P_GAIN"], P)
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_I_GAIN"], I)
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_D_GAIN"], D)
    
    def write_homing_offset(self, ID, val):
        self._write_byte(4, ID,
            self.protocol["control_table"]["ADDR_HOMING_OFFSET"], val)
        
    def write_position_limits(self, ID, minVal, maxVal):
        self._write_byte(4, ID,
            self.protocol["control_table"]["ADDR_MAX_POSITION_LIMIT"], maxVal)
        self._write_byte(4, ID,
            self.protocol["control_table"]["ADDR_MIN_POSITION_LIMIT"], minVal)
        # print(f"Set position limit to [{minVal}, {maxVal}]")
    
    def close_to_goal(self, ID, goal, delta=20):
        cur_pos = self.read_position(ID)
        if abs(cur_pos - goal) < delta:
            return True
        return False

    def write_goal_current(self, ID, cur):
        # set current control
        # self.set_control_mode(ID, current=True, enable_torque=True)
        # cur = np.clip(cur, *self.protocol["workspace"]["max_current"])
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_GOAL_CURRENT"], cur)
    
    def write_PID(self, ID):
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_P_GAIN"], 
            self.protocol["workspace"]["PID"][0])
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_I_GAIN"], 
            self.protocol["workspace"]["PID"][1])
        self._write_byte(2, ID, 
            self.protocol["control_table"]["ADDR_POSITION_D_GAIN"], 
            self.protocol["workspace"]["PID"][2])
    
    def LED(self, ID, on=True):
        self._write_byte(1, ID, 
            self.protocol["control_table"]["ADDR_LED"], 1 if on else 0)
    
    def home_gripper(self):
        """
        Home the gripper, update the range by first closing the gripper
        then opening the gripper.
        """
        self.disable_torque()
        self.set_control_mode(self.gripper_id, "current")

        # open the gripper with current control
        self.enable_torque()
        self.write_goal_current(self.gripper_id, -self.protocol["device"]["HOMING_CURRENT"])
        diff = 1000
        last_position = None
        # print("Homing. Opening gripper...")
        while diff > 3:
            time.sleep(0.3)
            cur_position = self.read_position(self.gripper_id)
            if last_position is not None:
                diff = abs(cur_position - last_position)
            last_position = cur_position
        # print("Gripper is at maximum")
        self.disable_torque()
        # set HOMING_OFFSET
        cur_offset = self.read_homing_offset(self.gripper_id)
        homing_offset = cur_offset - last_position
        # print(homing_offset, cur_offset, last_position)
        self.write_homing_offset(self.gripper_id, homing_offset)
        # print(f"Set homing offset as {homing_offset}")

        # close the gripper with current control
        self.enable_torque()
        self.write_goal_current(self.gripper_id, self.protocol["device"]["HOMING_CURRENT"])
        diff = 1000
        last_position = None
        # print("Homing. Closing gripper...")
        while diff > 10:
            time.sleep(0.1)
            cur_position = self.read_position(self.gripper_id)
            if last_position is not None:
                diff = abs(cur_position - last_position)
            last_position = cur_position
        # print("Gripper is at minimum")
        self.disable_torque()
        # set position limits
        self.write_position_limits(self.gripper_id, 0, last_position)

        self._gripper_range = last_position
        self._homed_gripper = True
        print("Homing sequence finished. Range: {} steps or {} degrees".format(
            self._gripper_range, self.position_to_deg(self._gripper_range)))
    
    def start(self, gripper_only=False):
        assert self._homed_gripper
        self.set_control_mode(self.gripper_id, "cbp")

        self.enable_torque()

        self.write_goal_pos(self.gripper_id, 0)
        self.write_goal_current(self.gripper_id, self.protocol["device"]["CLOSING_CURRENT"])
        time.sleep(1)
    
    @staticmethod
    def cur_feedback_curve(val):
        return int((abs(val) / 40.) ** 3 * 40) + 10

    def goto_gripper(self, pos):
        assert pos >= 0 and pos <= 1, "Pos must be between 0 and 1"
        des_gripper_pos = int((1-pos) * self._gripper_range)
        self.write_goal_pos(self.gripper_id, des_gripper_pos)
    
    def open(self, cur=50):
        self.write_goal_current(self.gripper_id, int(cur))
        self.goto_gripper(1)

    def close(self, cur=50):
        self.write_goal_current(self.gripper_id, int(cur))
        self.goto_gripper(0)

    def __del__(self):
        self.open()
        time.sleep(1)
        self.disable_torque()
        print("Gripper disabled.")
    
if __name__ == "__main__":
    gripper = DxlGripperInterface()
    gripper.init()
    gripper.home_gripper()

    gripper.start()
        # time.sleep(0.01)
    # gripper.write_goal_current(gripper.gripper_id, 50)
    # gripper.enable_torque(True, False)